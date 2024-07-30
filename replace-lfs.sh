#!/bin/bash

# Find all LFS pointers and replace with actual files
git ls-tree -r -z --name-only HEAD | while IFS= read -r -d '' file; do
  if [ -f "$file" ]; then
    # Check if the file is an LFS pointer
    if grep -q "oid sha256:" "$file"; then
      echo "Replacing LFS pointer: $file"
      git cat-file -p "HEAD:$file" > tmpfile
      mv tmpfile "$file"
      git add "$file"
    fi
  fi
done

g
git commit -m "Replace LFS pointers with actual files"