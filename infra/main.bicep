// Infraestrutura do assistente no Azure (escopo: grupo de recursos).
//
//   Container Apps (API, escala a zero) ──► PostgreSQL Flexible + pgvector
//        │  identidade gerenciada                    ▲
//        ├──► Key Vault (chave do Gemini, API key, URL do banco)
//        └──► Application Insights (traces OpenTelemetry) ◄── Log Analytics
//
// Pensado para o plano gratuito / Azure for Students: B1ms no Postgres, consumo no Container
// Apps e retenção mínima de logs. Detalhes em docs/estudo/04-nuvem-azure.md.

targetScope = 'resourceGroup'

@description('Região. A assinatura Azure for Students só permite algumas; canadacentral é a mais próxima do Brasil entre elas.')
param location string = 'canadacentral'

@description('Imagem do container (GitHub Container Registry).')
param containerImage string

@secure()
@description('Chave da API do Gemini.')
param geminiApiKey string

@secure()
@description('Chave exigida no cabeçalho X-API-Key das rotas /ingest, /search e /chat.')
param apiKey string

@secure()
@description('Senha do administrador do Postgres (só letras e números, para caber na URL).')
param dbAdminPassword string

@description('Object ID de quem administra (pode ler e gravar segredos no Key Vault). Vazio = ninguém além da app.')
param adminPrincipalId string = ''

// O GitHub manda o "subject" do token OIDC com IDs imutáveis: repo:dono@ID/nome@ID:ref:...
// Achado no 1º deploy (AADSTS700213). Com os IDs, um repositório recriado com o mesmo nome
// não herda a confiança. IDs: gh api users/<dono> --jq .id e gh api repos/<dono>/<nome> --jq .id
@description('Subject do token OIDC do GitHub autorizado a fazer deploy (só a main deste repositório).')
param githubOidcSubject string = 'repo:giuliagarbelini26296@218857421/personal-ai-assistant-rag-mcp@1464762314:ref:refs/heads/main'

@description('Embedder: hash (local, sem cota) ou gemini (melhor busca, cota grátis de 1.000/dia). Trocar exige reindexar: o banco não mistura embedders.')
@allowed([ 'hash', 'gemini' ])
param embeddingProvider string = 'hash'

var suffix = uniqueString(resourceGroup().id)
var dbAdmin = 'assistant'
var dbName = 'assistant'

// ------------------------------------------------------------------ observabilidade
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-assistant-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: 1 } // teto diário: protege o crédito de um pico de logs
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-assistant-${suffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

// ------------------------------------------------------------------ identidade e segredos
resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-assistant-${suffix}'
  location: location
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-asst-${suffix}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true // acesso por papéis (RBAC), não por políticas de acesso
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource secretGemini 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'gemini-api-key'
  properties: { value: geminiApiKey }
}

resource secretApiKey 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'api-key'
  properties: { value: apiKey }
}

resource secretDbUrl 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'database-url'
  properties: {
    value: 'postgresql://${dbAdmin}:${dbAdminPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/${dbName}?sslmode=require'
  }
}

// "Key Vault Secrets User": a identidade da app só LÊ segredos (menor privilégio).
var kvSecretsUserRole = '4633458b-17de-408a-b874-0445c86b69e6' // gitleaks:allow (ID público do papel)
resource vaultAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: vault
  name: guid(vault.id, identity.id, kvSecretsUserRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRole)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// "Key Vault Secrets Officer" para a pessoa administradora: com RBAC, nem a dona da assinatura
// lê segredos sem um papel explícito.
var kvSecretsOfficerRole = 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7' // gitleaks:allow (ID público do papel)
resource vaultAdmin 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(adminPrincipalId)) {
  scope: vault
  name: guid(vault.id, adminPrincipalId, kvSecretsOfficerRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsOfficerRole)
    principalId: adminPrincipalId
    principalType: 'User'
  }
}

// Identidade usada pelo GitHub Actions para o deploy. Sem senha: o Azure confia no token OIDC
// que o GitHub emite SÓ para pushes na main deste repositório (campo subject).
resource deployIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-github-deploy-${suffix}'
  location: location
}

resource deployFederation 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: deployIdentity
  name: 'github-main'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: githubOidcSubject
    audiences: [ 'api://AzureADTokenExchange' ]
  }
}

// Contributor só neste grupo de recursos: o deploy não enxerga o resto da assinatura.
var contributorRole = 'b24988ac-6180-42a0-ab88-20f7382dd24c' // gitleaks:allow (ID público do papel)
resource deployAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(resourceGroup().id, deployIdentity.id, contributorRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRole)
    principalId: deployIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ------------------------------------------------------------------ banco
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2022-12-01' = {
  name: 'psql-assistant-${suffix}'
  location: location
  sku: { name: 'Standard_B1ms', tier: 'Burstable' } // incluído nos 12 meses gratuitos
  properties: {
    version: '16'
    administratorLogin: dbAdmin
    administratorLoginPassword: dbAdminPassword
    storage: { storageSizeGB: 32 }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
  }
}

// O Azure só permite extensões liberadas nesta configuração: sem ela, CREATE EXTENSION vector falha.
resource pgExtensions 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2022-12-01' = {
  parent: postgres
  name: 'azure.extensions'
  properties: { value: 'VECTOR', source: 'user-override' }
}

resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2022-12-01' = {
  parent: postgres
  name: dbName
  dependsOn: [ pgExtensions ]
}

// 0.0.0.0 = "serviços do Azure". O Container Apps de consumo não tem IP fixo; a alternativa
// fechada (rede virtual + endpoint privado) custa mais. O banco continua exigindo TLS e senha.
resource pgFirewallAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2022-12-01' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: { startIpAddress: '0.0.0.0', endIpAddress: '0.0.0.0' }
}

// ------------------------------------------------------------------ aplicação
resource environment 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: 'cae-assistant-${suffix}'
  location: location
  properties: {
    // Sem isto, o Azure cria o ambiente no modo "Express" (padrão novo), que NÃO aceita
    // segredos do Key Vault. Achado no 1º deploy: erro ExpressKeyVaultSecretsNotSupported.
    environmentMode: 'WorkloadProfiles'
    workloadProfiles: [ { name: 'Consumption', workloadProfileType: 'Consumption' } ]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-assistant'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [ vaultAccess, database, pgFirewallAzure ]
  properties: {
    managedEnvironmentId: environment.id
    workloadProfileName: 'Consumption'
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false // só HTTPS
      }
      // Os segredos ficam no Key Vault; a app guarda só a referência e lê com a identidade.
      secrets: [
        { name: 'gemini-api-key', keyVaultUrl: secretGemini.properties.secretUri, identity: identity.id }
        { name: 'api-key', keyVaultUrl: secretApiKey.properties.secretUri, identity: identity.id }
        { name: 'database-url', keyVaultUrl: secretDbUrl.properties.secretUri, identity: identity.id }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'VECTOR_STORE', value: 'pgvector' }
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'EMBEDDING_PROVIDER', value: embeddingProvider }
            { name: 'LLM_PROVIDER', value: 'gemini' }
            { name: 'GEMINI_API_KEY', secretRef: 'gemini-api-key' }
            { name: 'API_KEY', secretRef: 'api-key' }
            { name: 'OTEL_EXPORTER', value: 'azure' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
          ]
        }
      ]
      // Escala a zero sem uso (sem custo parado); 1 réplica basta para portfólio.
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}

output apiUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output containerAppName string = app.name
output keyVaultName string = vault.name
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output deployClientId string = deployIdentity.properties.clientId
