#!/usr/bin/env python3
"""E2E test for contextplus-rs MCP server via stdio transport."""

import json
import subprocess
import sys
import time
import os

BINARY = "/tmp/contextplus-rs/target/release/contextplus-rs"
ROOT = "/workspace"

os.environ["OLLAMA_HOST"] = "http://172.18.0.1:11434"
os.environ["OLLAMA_EMBED_MODEL"] = "snowflake-arctic-embed2"
os.environ["OLLAMA_CHAT_MODEL"] = "llama3.2"

def main():
    proc = subprocess.Popen(
        [BINARY, "--root-dir", ROOT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    results = {}
    errors = []

    def send(msg):
        line = json.dumps(msg)
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def read_response(timeout=120):
        """Read one line from stdout (JSON-RPC response)."""
        import select
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if ready:
            line = proc.stdout.readline()
            if line:
                return json.loads(line.strip())
        return None

    # 1. Initialize
    print(">>> Sending initialize...")
    send({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0.0"}
        }
    })
    resp = read_response(10)
    if resp:
        print(f"<<< initialize: OK (server={resp.get('result', {}).get('serverInfo', {}).get('name', 'unknown')})")
        results["initialize"] = "PASS"
    else:
        print("<<< initialize: TIMEOUT")
        results["initialize"] = "FAIL"
        errors.append("initialize timed out")

    # 2. Initialized notification
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    time.sleep(0.2)

    # 3. List tools
    print(">>> Sending tools/list...")
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = read_response(5)
    if resp and "result" in resp:
        tools = resp["result"].get("tools", [])
        tool_names = [t["name"] for t in tools]
        print(f"<<< tools/list: {len(tools)} tools found")
        for name in sorted(tool_names):
            print(f"    - {name}")
        results["tools_list"] = "PASS" if len(tools) >= 15 else "FAIL"
    else:
        print("<<< tools/list: FAIL")
        results["tools_list"] = "FAIL"
        errors.append("tools/list failed")

    # 4. get_context_tree
    print("\n>>> Calling get_context_tree (depthLimit=1, maxTokens=3000)...")
    send({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "get_context_tree", "arguments": {
            "rootDir": ROOT, "depthLimit": 1, "maxTokens": 3000
        }}
    })
    resp = read_response(30)
    if resp and "result" in resp:
        content = resp["result"].get("content", [{}])
        text = content[0].get("text", "") if content else ""
        print(f"<<< get_context_tree: {len(text)} chars")
        print(f"    Preview: {text[:200]}...")
        results["context_tree"] = "PASS" if len(text) > 50 else "FAIL"
    else:
        print(f"<<< get_context_tree: FAIL ({resp})")
        results["context_tree"] = "FAIL"
        errors.append("get_context_tree failed")

    # 5. get_file_skeleton
    print("\n>>> Calling get_file_skeleton...")
    send({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "get_file_skeleton", "arguments": {
            "filePath": "packages/domains/forms/service/form-service.ts",
            "rootDir": ROOT
        }}
    })
    resp = read_response(10)
    if resp and "result" in resp:
        content = resp["result"].get("content", [{}])
        text = content[0].get("text", "") if content else ""
        print(f"<<< get_file_skeleton: {len(text)} chars")
        print(f"    Preview: {text[:200]}...")
        results["file_skeleton"] = "PASS" if len(text) > 20 else "FAIL"
    else:
        print(f"<<< get_file_skeleton: FAIL ({resp})")
        results["file_skeleton"] = "FAIL"
        errors.append("get_file_skeleton failed")

    # 6. semantic_code_search (USES REAL OLLAMA!)
    # Use a small subdirectory to avoid embedding 1000+ files
    SEARCH_ROOT = ROOT + "/packages/domains/forms"
    print(f"\n>>> Calling semantic_code_search on {SEARCH_ROOT} (uses Ollama embeddings)...")
    send({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "semantic_code_search", "arguments": {
            "query": "form validation and submission",
            "rootDir": SEARCH_ROOT, "topK": 3
        }}
    })
    resp = read_response(120)  # May take a while for cold embed
    if resp and "result" in resp:
        content = resp["result"].get("content", [{}])
        text = content[0].get("text", "") if content else ""
        is_error = resp["result"].get("isError", False)
        if is_error:
            print(f"<<< semantic_code_search: ERROR")
            print(f"    {text[:500]}")
            results["semantic_search"] = "ERROR"
            errors.append(f"semantic_code_search error: {text[:200]}")
        else:
            print(f"<<< semantic_code_search: {len(text)} chars")
            print(f"    Preview: {text[:300]}...")
            results["semantic_search"] = "PASS"
    else:
        print(f"<<< semantic_code_search: TIMEOUT")
        results["semantic_search"] = "TIMEOUT"
        errors.append("semantic_code_search timed out")

    # 7. get_blast_radius
    print("\n>>> Calling get_blast_radius...")
    send({
        "jsonrpc": "2.0", "id": 6, "method": "tools/call",
        "params": {"name": "get_blast_radius", "arguments": {
            "symbolName": "createFormService",
            "rootDir": ROOT
        }}
    })
    resp = read_response(30)
    if resp and "result" in resp:
        content = resp["result"].get("content", [{}])
        text = content[0].get("text", "") if content else ""
        print(f"<<< get_blast_radius: {len(text)} chars")
        print(f"    Preview: {text[:200]}...")
        results["blast_radius"] = "PASS" if len(text) > 10 else "FAIL"
    else:
        print(f"<<< get_blast_radius: FAIL ({resp})")
        results["blast_radius"] = "FAIL"
        errors.append("get_blast_radius failed")

    # Summary
    print("\n" + "=" * 60)
    print("E2E TEST RESULTS")
    print("=" * 60)
    for test, status in results.items():
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"  {icon} {test}: {status}")

    passed = sum(1 for s in results.values() if s == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} passed")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    # Clean up
    proc.stdin.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    stderr = proc.stderr.read()
    if stderr:
        print(f"\nServer stderr (last 500 chars):\n{stderr[-500:]}")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
