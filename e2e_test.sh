#!/usr/bin/env bash
# E2E test: sends MCP JSON-RPC messages to the contextplus-rs binary via stdin/stdout
set -euo pipefail

BINARY="/tmp/contextplus-rs/target/release/contextplus-rs"
ROOT="/workspace"
export OLLAMA_HOST="http://172.18.0.1:11434"
export OLLAMA_EMBED_MODEL="snowflake-arctic-embed2"
export OLLAMA_CHAT_MODEL="llama3.2"

# Helper: send a JSON-RPC message and read the response
send_rpc() {
    local msg="$1"
    echo "$msg"
}

# Build the full MCP conversation
{
    # 1. Initialize
    send_rpc '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"e2e-test","version":"1.0.0"}}}'

    # 2. Initialized notification
    send_rpc '{"jsonrpc":"2.0","method":"notifications/initialized"}'

    # 3. List tools
    send_rpc '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

    # 4. Call get_context_tree
    send_rpc '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_context_tree","arguments":{"rootDir":"'"$ROOT"'","depthLimit":2,"maxTokens":5000}}}'

    # 5. Call get_file_skeleton on a known file
    send_rpc '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_file_skeleton","arguments":{"filePath":"packages/domains/forms/service/form-service.ts","rootDir":"'"$ROOT"'"}}}'

    # 6. Call semantic_code_search (uses Ollama embeddings!)
    send_rpc '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"semantic_code_search","arguments":{"query":"authentication and authorization","rootDir":"'"$ROOT"'","topK":3}}}'

    # 7. Call get_blast_radius
    send_rpc '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"get_blast_radius","arguments":{"symbolName":"createFormService","rootDir":"'"$ROOT"'"}}}'

    # 8. Call run_static_analysis
    send_rpc '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"run_static_analysis","arguments":{"rootDir":"'"$ROOT"'"}}}'

    # Wait for responses then close
    sleep 120
} | timeout 180 "$BINARY" --root-dir "$ROOT" 2>/tmp/contextplus-e2e-stderr.log | while IFS= read -r line; do
    echo "$line"
done
