I'm building a tool called BrandBench that uses OpenAI's Responses API with web search. Here's my current implementation:

@src/benchmark/prompt_executor.py

The issue is that I'm getting "url_citation" errors for most requests when web search is enabled. The code tries to extract cited URLs from the response like this:

```python
if enable_web_search and output:
    cited_urls = []
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    annotations = content.get("annotations", [])
                    for ann in annotations:
                        if ann.get("type") == "url_citation":
                            cited_urls.append(ann["url_citation"]["url"])
    result["cited_urls"] = cited_urls
```

Please help me:
1. Understand why I'm getting url_citation errors
2. Fix the code to properly handle web search responses
3. Ensure the code works with the latest OpenAI Responses API format

Include any necessary changes to handle the response structure correctly.

---

## 1 — Why the “url_citation” KeyError happens

Beginning with the **March 11 2025** rollout of the Responses API & hosted **`web_search`** tool, the shape of an _annotation object_ changed:

Old (late-2024 “preview”)  
```json
{
  "type": "url_citation",
  "url_citation": {          // <— nested object
    "url":   "...",
    "title": "...",
    ...
  }
}
```

Current (2025-03-11 and later)  
```json
{
  "type":  "url_citation",
  "url":   "...",
  "title": "...",
  "index": null              // start/end indexes were collapsed to one field
}
```  
Source: OpenAI cookbook example “Web Search and States” published 2025-03-11 ([cookbook.openai.com](https://cookbook.openai.com/examples/responses_api/responses_example))

Your extractor still expects the **nested** `url_citation` key, so it raises a `KeyError` for most responses that use the new flat structure. The exception bubbles up and looks like an “url_citation error”.

---

## 2 — Patch: make the extractor tolerant of both formats

```python
# src/benchmark/prompt_executor.py
# ⋮

def _collect_cited_urls(output: List[dict]) -> List[str]:
    """Return a list of cited URLs, compatible with old and new annotation formats."""
    urls: list[str] = []

    for item in output:
        if item.get("type") != "message":
            continue

        for chunk in item.get("content", []):
            if chunk.get("type") != "output_text":
                continue

            for ann in chunk.get("annotations", []):
                if ann.get("type") != "url_citation":
                    continue

                # ❶ New schema (flat)
                url = ann.get("url")

                # ❷ Old preview schema (nested object)
                if url is None and isinstance(ann.get("url_citation"), dict):
                    url = ann["url_citation"].get("url")

                if url:
                    urls.append(url)

    # de-dupe & keep stable order
    return list(dict.fromkeys(urls))
```

Then replace the old loop:

```python
if enable_web_search and output:
    result["cited_urls"] = _collect_cited_urls(output)
else:
    result["cited_urls"] = []
```

That’s the only change needed to stop the KeyError.

---

## 3 — Other adjustments for the latest API format

1. **Tool name** – `"web_search"` is now the stable name, but `"web_search_preview"` is still aliased. You can future-proof:

   ```python
   if enable_web_search:
       kwargs["tools"] = [{"type": "web_search"}]  # or make it configurable
   ```

2. **Output item types** – Alongside `"message"` you may also see `"web_search_call"` blocks. You can capture metadata if you care:

   ```python
   search_calls = [b for b in output if b.get("type") == "web_search_call"]
   ```

3. **Safety / rate-limits** – Hosted tools count against your request+context tokens. Avoid `search_context_size: "high"` unless you need it.

4. **UI requirement** – OpenAI policy (updated Feb 2025) requires every cited link to be visible & clickable to end-users.

---

## 4 — Full minimal diff (for Claude)

<details><summary>patch</summary>

```diff
@@
         # Extract web search information if available
-        if enable_web_search and output:
-            cited_urls = []
-            for item in output:
-                if item.get("type") == "message":
-                    for content in item.get("content", []):
-                        if content.get("type") == "output_text":
-                            annotations = content.get("annotations", [])
-                            for ann in annotations:
-                                if ann.get("type") == "url_citation":
-                                    cited_urls.append(ann["url_citation"]["url"])
-            result["cited_urls"] = cited_urls
-        else:
-            result["cited_urls"] = []
+        if enable_web_search and output:
+            result["cited_urls"] = _collect_cited_urls(output)
+        else:
+            result["cited_urls"] = []
```
</details>

---

## 5 — Potential pitfalls & security notes

* **Schema drift** – The Responses API is still marked _beta_; expect more renames. Keep extractor logic **schema-agnostic** (e.g. fallbacks like above).
* **Unhandled `None`** – `index` can be `null`; don’t cast directly to `int`.
* **Duplicate URLs** – The same link can appear multiple times in one answer; de-duping avoids redundant logging.
* **Hostile links** – Consider validating each URL (length ≤ 2 kB, scheme `https`) before persisting or showing it.
* **Rate limiting** – Every tool call counts as an additional request inside your one outer call; watch your org’s RPM if you parallelize heavily.

---

### References

1. OpenAI cookbook – “Web Search and States with Responses API”, example output shows flat annotation ​structure (Mar 11 2025) ([cookbook.openai.com](https://cookbook.openai.com/examples/responses_api/responses_example))  
2. OpenAPI community schema snapshot (showing flat UrlCitation definition) ([gist.github.com](https://gist.github.com/lrtherond/8287316203ea8d4ff5808067b4d00560?utm_source=chatgpt.com))

You can now re-run BrandBench with `enable_web_search=True`; the KeyError should disappear and `result["cited_urls"]` will be populated correctly.