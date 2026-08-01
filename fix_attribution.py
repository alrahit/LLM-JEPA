content = open("docs/full_documentation.md", encoding="utf-8").read()
old = "**Researcher:** Al Rahit"
new = "**Research led by:** Gollam Rabby, Junior Research Group Leader (Nachwuchsgruppenleiter)\n**Contributor:** Al Rahit"
count = content.count(old)
assert count == 1, f"Expected 1 match, found {count} - aborting"
content = content.replace(old, new)
open("docs/full_documentation.md", "w", encoding="utf-8").write(content)
print("Patched successfully, encoding preserved")
