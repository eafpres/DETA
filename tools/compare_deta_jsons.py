#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:02:42 2026

@author: eafpres
"""

from pathlib import Path
import argparse
import csv
import ijson
def iter_image_file_names(json_path):
  """Yield image file names from a COCO/DETA JSON file.
  Args:
    json_path: Path to a JSON file.
  Yields:
    File names from the top-level images array.
  """
  json_path = Path(json_path)
  with json_path.open('rb') as f:
    for image in ijson.items(f, 'images.item'):
      file_name = image.get('file_name')
      if file_name is not None:
        yield str(file_name)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--v3-json', required = True)
  parser.add_argument('--v4-json', required = True)
  parser.add_argument('--out-txt', default = 'v4_not_in_v3.txt')
  parser.add_argument('--out-csv', default = 'v4_not_in_v3.csv')
  args = parser.parse_args()
#
# Build a filename set from v3 only.
#
  v3_files = set(iter_image_file_names(args.v3_json))
#
# Stream v4 and keep files that are absent from v3.
#
  missing_from_v3 = []
  seen_v4 = set()
  for file_name in iter_image_file_names(args.v4_json):
    if file_name in seen_v4:
      continue
    seen_v4.add(file_name)
    if file_name not in v3_files:
      missing_from_v3.append(file_name)
#
# Write plain text output.
#
  out_txt = Path(args.out_txt)
  out_txt.write_text(
    '\n'.join(missing_from_v3) + ('\n' if missing_from_v3 else ''),
    encoding = 'utf-8'
  )
#
# Write CSV output.
#
  out_csv = Path(args.out_csv)
  with out_csv.open('w', newline = '', encoding = 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name'])
    for file_name in missing_from_v3:
      writer.writerow([file_name])
  print(f'v3 unique image files: {len(v3_files):,}')
  print(f'v4 unique image files: {len(seen_v4):,}')
  print(f'v4 files not in v3: {len(missing_from_v3):,}')
  print(f'wrote: {out_txt}')
  print(f'wrote: {out_csv}')
if __name__ == '__main__':
  main()
