# ShardGuard


## Setup

```bash
pip install poetry
poetry install
cp .env.example .env   # add your OPENAI_API_KEY and any MCP auth keys
```

## Run

```bash
# See plan only (no tool execution)
poetry run shardguard plan "Get the HR record for jane@example.com"

# Plan and execute
poetry run shardguard plan "Get the HR record for jane@example.com" -x

# Run without ShardGuard (baseline)
poetry run shardguard baseline "Get the HR record for jane@example.com"
```

## List available MCP tools

```bash
# Summary view
poetry run shardguard list-tools

# Verbose (shows descriptions)
poetry run shardguard list-tools --verbose

# Against a specific registry
poetry run shardguard list-tools --registry-path path/to/mcp_registry.json
```

## Manage MCP servers

The registry file (`src/shardguard/mcp_servers/mcp_registry.json`) defines which MCP servers are available.

### Add an HTTP MCP server

```bash
poetry run shardguard registry add-mcp \
  --registry src/shardguard/mcp_servers/mcp_registry.json \
  --name my-service \
  --transport streamable-http \
  --url https://my-service.example.com/mcp \
  --desc "My service: look up records by name"
```

### Add a stdio MCP server

```bash
poetry run shardguard registry add-mcp \
  --registry src/shardguard/mcp_servers/mcp_registry.json \
  --name github-reader \
  --transport stdio \
  --cmd npx \
  --args '"-y","@modelcontextprotocol/server-github"'
```

### Remove one or more MCP servers

```bash
poetry run shardguard registry remove-mcp \
  --registry src/shardguard/mcp_servers/mcp_registry.json \
  my-service

# Remove multiple at once
poetry run shardguard registry remove-mcp \
  --registry src/shardguard/mcp_servers/mcp_registry.json \
  my-service github-reader
```
