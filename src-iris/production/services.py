"""WikiGraph - business service that polls GraphKB.ChatRequest for new questions.

The WSGI app only inserts a 'pending' ChatRequest row; this polling service picks it up and
drives the pipeline. This deliberately avoids Ens.Director.CreateBusinessService from the web
workers, which registers the worker's job in ^Ens.JobStatus and races across workers
(<Ens>ErrJobRegistryNotClean). The poller is a normal production job, so there is no such race.
"""

import json

from intersystems_pyprod import IRISParameter, IRISProperty, InboundAdapter, BusinessService, Status

from common import logErrorAndReturnStatus
from messages import chatRequestMsg, resolveEntitiesMsg

iris_package_name = "WikiGraph"


class chatPollAdapter(InboundAdapter):

    def OnTask(self):
        status = Status.OK()
        try:
            import iris
            pending = []
            # 'pending' = a chat question; 'ingest' = add a Wikipedia article by title;
            # 'resolve' = dedup the knowledge graph (question holds the dryRun flag "1"/"0")
            for row in iris.sql.exec(
                    "SELECT TOP 5 %ID, question, lang, status FROM GraphKB.ChatRequest "
                    "WHERE status IN ('pending', 'ingest', 'resolve') ORDER BY %ID"):
                pending.append((int(row[0]), row[1], row[2], row[3]))
            for requestId, question, lang, rowStatus in pending:
                if rowStatus == "ingest":
                    mode, detail = "ingest", "Adding the article"
                elif rowStatus == "resolve":
                    mode, detail = "resolve", "Resolving duplicate entities"
                else:
                    mode, detail = "chat", "Searching the knowledge base"
                # claim the row so the next poll won't pick it up again
                iris.sql.exec(
                    "UPDATE GraphKB.ChatRequest SET status = 'searching', statusDetail = ?, "
                    "updatedAt = CURRENT_TIMESTAMP WHERE %ID = ? AND status = ?", detail, requestId, rowStatus)
                self.business_host_process_input({"requestId": requestId, "question": question, "lang": lang, "mode": mode})
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in chatPollAdapter OnTask: " + str(e))
        return status


class chatService(BusinessService):

    ADAPTER: str = IRISParameter(value="WikiGraph.chatPollAdapter", description="Polling inbound adapter")
    targetConfigName = IRISProperty(
        default="chatProcess",
        description="Business process that orchestrates the chat pipeline",
        settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")
    resolveTarget = IRISProperty(
        default="resolutionOperation",
        description="Business operation that dedups the knowledge graph",
        settings="Target:selector?context={Ens.ContextSearch/ProductionItems?targets=1&productionName=@productionId}")

    def OnProcessInput(self, input):
        status = Status.OK()
        try:
            payload = json.loads(input) if isinstance(input, str) else input
            requestId = int(payload["requestId"])
            lang = payload.get("lang", "en")
            if payload.get("mode") == "resolve":
                # 'question' carries the resolve params as JSON: {dryRun, exclude}
                try:
                    params = json.loads(payload.get("question") or "{}")
                except ValueError:
                    params = {}
                dryRun = 1 if params.get("dryRun", True) else 0
                excludeJson = json.dumps(params.get("exclude") or [])
                status = self.SendRequestAsync(self.resolveTarget,
                                               resolveEntitiesMsg(requestId, lang, 0, dryRun, excludeJson))
            else:
                message = chatRequestMsg(requestId, payload["question"], lang, payload.get("mode", "chat"))
                status = self.SendRequestAsync(self.targetConfigName, message)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in chatService OnProcessInput: " + str(e))
        return status
