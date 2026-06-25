Class App.Installer
{

Parameter NAMESPACE = "GRAPHRAG";

/// Name of the %Embedding.Config entry referenced by the EMBEDDING columns.
Parameter EMBEDDINGCONFIG = "wikiembed";

/// Dummy SSL config name (required by %Embedding.OpenAI validation; unused because Https=0).
Parameter OLLAMASSL = "ollama";

Parameter WEBAPP = "/wikigraph";

XData setup
{
<Manifest>
  <Default Name="Namespace" Value="GRAPHRAG"/>
  <Default Name="database" Value="graphrag"/>

  <Namespace Name="${Namespace}" Code="${Namespace}-CODE" Data="${Namespace}-DATA" Create="yes" Ensemble="1">
    <Configuration>
      <Database Name="${Namespace}-DATA" Dir="${mgrdir}${database}/data" Create="yes" Resource="%DB_${Namespace}-DATA"/>
      <Database Name="${Namespace}-CODE" Dir="${mgrdir}${database}/code" Create="yes" Resource="%DB_${Namespace}-CODE"/>
    </Configuration>
  </Namespace>
</Manifest>
}

ClassMethod setup(ByRef pVars, pLogLevel As %Integer = 3, pInstaller As %Installer.Installer, pLogger As %Installer.AbstractLogger) As %Status [ CodeMode = objectgenerator, Internal ]
{
  #; Let XGL document generate code for this method.
  Quit ##class(%Installer.Manifest).%Generate(%compiledclass, %code, "setup")
}

/// Post-setup configuration: WSGI web app, EMBEDDING config, GraphRAG schema.
ClassMethod configure() As %Status
{
  Set sc = $$$OK
  Try {
    Write !, "Creating WSGI web application..."
    $$$ThrowOnError(..createWebApp())
    Write !, "Creating EMBEDDING configuration..."
    $$$ThrowOnError(..createEmbeddingConfig())
    Write !, "Creating GraphKB schema..."
    $$$ThrowOnError(..createSchema())
    Write !, "Configuration complete.", !
  }
  Catch ex {
    Set sc = ex.AsStatus()
    Do $SYSTEM.Status.DisplayError(sc)
  }
  Quit sc
}

/// Registers the Flask chatbot as a WSGI web application hosted by IRIS.
ClassMethod createWebApp() As %Status
{
  New $NAMESPACE
  Set $NAMESPACE = "%SYS"
  Quit:##class(Security.Applications).Exists(..#WEBAPP) $$$OK
  Set props("NameSpace") = ..#NAMESPACE
  Set props("Enabled") = 1
  Set props("AutheEnabled") = 64
  Set props("MatchRoles") = ":%All"
  Set props("DispatchClass") = "%SYS.Python.WSGI"
  Set props("WSGIAppLocation") = "/opt/irisbuild/wsgi"
  Set props("WSGIAppName") = "app"
  Set props("WSGICallable") = "app"
  Quit ##class(Security.Applications).Create(..#WEBAPP, .props)
}

/// Registers the EMBEDDING datatype config: %Embedding.OpenAI pointed at the Ollama
/// container's OpenAI-compatible /v1/embeddings endpoint. IRIS never loads a model
/// in-process - each EMBEDDING() is a quick HTTP call to Ollama.
ClassMethod createEmbeddingConfig() As %Status
{
  // %Embedding.OpenAI.IsValidConfig requires an existing SSL config name, even though we
  // talk plain HTTP to Ollama (httpConfig sets Https=0). Create a dummy one.
  New $NAMESPACE
  Set $NAMESPACE = "%SYS"
  If '##class(Security.SSLConfigs).Exists(..#OLLAMASSL) {
    $$$ThrowOnError(##class(Security.SSLConfigs).Create(..#OLLAMASSL))
  }
  Set $NAMESPACE = ..#NAMESPACE

  Set embedModel = $SYSTEM.Util.GetEnviron("EMBEDDING_MODEL")
  Set:embedModel="" embedModel = "leoipulsar/harrier-0.6b:latest"
  Set ollamaHost = $SYSTEM.Util.GetEnviron("OLLAMA_HOST")
  Set:ollamaHost="" ollamaHost = "ollama"
  Set ollamaPort = $SYSTEM.Util.GetEnviron("OLLAMA_EMBED_PORT")
  Set:ollamaPort="" ollamaPort = 11434

  Set rs = ##class(%SQL.Statement).%ExecDirect(, "SELECT COUNT(*) AS Total FROM %Embedding.Config WHERE Name = ?", ..#EMBEDDINGCONFIG)
  If rs.%SQLCODE < 0 Quit $$$ERROR($$$GeneralError, "Embedding config check failed: "_rs.%Message)
  Do rs.%Next()
  Quit:rs.Total>0 $$$OK

  Set config = {"modelName": (embedModel), "apiKey": "ollama", "sslConfig": (..#OLLAMASSL), "checkTokenCount": false}
  Set config.httpConfig = {"Server": (ollamaHost), "Https": 0}
  Set config.httpConfig.Port = +ollamaPort
  Set rs = ##class(%SQL.Statement).%ExecDirect(,
    "INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, VectorLength, Description) VALUES (?, ?, ?, ?, ?)",
    ..#EMBEDDINGCONFIG, config.%ToJSON(), "%Embedding.OpenAI", 1024,
    "Embedding model served by Ollama (OpenAI-compatible /v1/embeddings)")
  If rs.%SQLCODE < 0 Quit $$$ERROR($$$GeneralError, "Embedding config insert failed: "_rs.%Message)
  Quit $$$OK
}

/// Creates the GraphRAG knowledge-base tables and indexes. Idempotent.
ClassMethod createSchema() As %Status
{
  New $NAMESPACE
  Set $NAMESPACE = ..#NAMESPACE
  Set cfg = ..#EMBEDDINGCONFIG
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.Document (pageId INTEGER, title VARCHAR(500), lang VARCHAR(5), url VARCHAR(1000), ingestedAt TIMESTAMP)"
  Set ddl($Increment(ddl)) = "CREATE UNIQUE INDEX DocumentPageIdx ON GraphKB.Document (pageId, lang)"
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.Chunk (documentRef INTEGER, chunkIndex INTEGER, chunkText VARCHAR(30000), embedText VARCHAR(31000), chunkVector EMBEDDING('"_cfg_"','embedText'))"
  Set ddl($Increment(ddl)) = "CREATE INDEX HNSWChunkIdx ON TABLE GraphKB.Chunk (chunkVector) AS HNSW(Distance='Cosine')"
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.Entity (entityName VARCHAR(300), entityType VARCHAR(100), lang VARCHAR(5), documentRef INTEGER, description VARCHAR(4000), aliases VARCHAR(1000), descVector EMBEDDING('"_cfg_"','description'))"
  Set ddl($Increment(ddl)) = "CREATE INDEX HNSWEntityIdx ON TABLE GraphKB.Entity (descVector) AS HNSW(Distance='Cosine')"
  Set ddl($Increment(ddl)) = "CREATE INDEX EntityNameIdx ON GraphKB.Entity (entityName, entityType, lang)"
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.Relationship (sourceEntityRef INTEGER, targetEntityRef INTEGER, predicate VARCHAR(200), description VARCHAR(4000), chunkRef INTEGER)"
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.EntityMention (entityRef INTEGER, chunkRef INTEGER)"
  Set ddl($Increment(ddl)) = "CREATE TABLE GraphKB.ChatRequest (question VARCHAR(2000), lang VARCHAR(5), status VARCHAR(20), statusDetail VARCHAR(500), answer VARCHAR(30000), createdAt TIMESTAMP, updatedAt TIMESTAMP)"
  For i = 1:1:ddl {
    Set rs = ##class(%SQL.Statement).%ExecDirect(, ddl(i))
    // -201 = table already exists, -324 = index already exists
    If (rs.%SQLCODE < 0) && (rs.%SQLCODE '= -201) && (rs.%SQLCODE '= -324) {
      Return $$$ERROR($$$GeneralError, "DDL failed (SQLCODE "_rs.%SQLCODE_"): "_ddl(i)_" : "_rs.%Message)
    }
  }
  Quit $$$OK
}

}
