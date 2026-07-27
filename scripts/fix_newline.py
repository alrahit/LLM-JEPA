content = open("evaluate.py", encoding="utf-8").read()
broken = "f.write(json.dumps(result_record) + \"\n\")"
fixed = "f.write(json.dumps(result_record) + \"\\n\")"
count = content.count(broken)
assert count == 1, f"Expected 1 match, found {count} - aborting"
content = content.replace(broken, fixed)
open("evaluate.py", "w", encoding="utf-8").write(content)
print("Fixed successfully")
