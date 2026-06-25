"""WikiGraph - business service that polls GraphKB.ChatRequest for new questions.

The WSGI app only inserts a 'pending' ChatRequest row; this polling service picks it up and
drives the pipeline. This deliberately avoids Ens.Director.CreateBusinessService from the web
workers, which registers the worker's job in ^Ens.JobStatus and races across workers
(<Ens>ErrJobRegistryNotClean). The poller is a normal production job, so there is no such race.
"""

import json

from intersystems_pyprod import IRISParameter, IRISProperty, InboundAdapter, BusinessService, Status

from common import logErrorAndReturnStatus
from messages import chatRequestMsg

iris_package_name = "WikiGraph"


class chatPollAdapter(InboundAdapter):

    def OnTask(self):
        status = Status.OK()
        try:
            import iris
            pending = []
            # 'pending' = a chat question; 'ingest' = add a Wikipedia article by title
            for row in iris.sql.exec(
                    "SELECT TOP 5 %ID, question, lang, status FROM GraphKB.ChatRequest "
                    "WHERE status IN ('pending', 'ingest') ORDER BY %ID"):
                pending.append((int(row[0]), row[1], row[2], row[3]))
            for requestId, question, lang, rowStatus in pending:
                mode = "ingest" if rowStatus == "ingest" else "chat"
                detail = "Adding the article" if mode == "ingest" else "Searching the knowledge base"
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

    def OnProcessInput(self, input):
        status = Status.OK()
        try:
            payload = json.loads(input) if isinstance(input, str) else input
            message = chatRequestMsg(int(payload["requestId"]), payload["question"],
                                     payload.get("lang", "en"), payload.get("mode", "chat"))
            status = self.SendRequestAsync(self.targetConfigName, message)
        except Exception as e:
            status, _ = logErrorAndReturnStatus("ERROR in chatService OnProcessInput: " + str(e))
        return status
