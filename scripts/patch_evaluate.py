content = open("evaluate.py", encoding="utf-8").read()

old = """                    if equal:
                        sim_list.append(cos_sim)
                    else:
                        sim_list_untune.append(cos_sim)
                else:
                    sim_list.append(cos_sim)"""

new = """                    if equal:
                        sim_list.append(cos_sim)
                    else:
                        sim_list_untune.append(cos_sim)
                    result_record = {
                        "index": idx,
                        "ground_truth": messages[-1]["content"] if isinstance(messages, list) else None,
                        "generated": generated_response,
                        "correct": bool(equal),
                    }
                    results.append(result_record)
                    f.write(json.dumps(result_record) + "\n")
                    f.flush()
                else:
                    sim_list.append(cos_sim)"""

count = content.count(old)
assert count == 1, f"Expected 1 match, found {count} - aborting, no changes made"
content = content.replace(old, new)
open("evaluate.py", "w", encoding="utf-8").write(content)
print("Patched successfully - eval_*.jsonl will now save each example")
