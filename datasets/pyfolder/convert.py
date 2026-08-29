import json
from pathlib import Path

filename = input("Enter the filename (with extension) to convert: ")
outfilename = input("Enter the output filename (with no extension): ")

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / filename
JSONL_OUTPUT_FILE = BASE_DIR / f"{outfilename}.jsonl"
TXT_OUTPUT_FILE = BASE_DIR / f"{outfilename}.txt"


def read_txt_records(input_path: Path):
    records = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "|" not in line:
                skipped += 1
                print(f"[Line {line_number}] Missing '|', skipped.")
                continue

            user_input, output = line.split("|", 1)
            user_input = user_input.strip()
            output = output.strip()

            if not user_input or not output:
                skipped += 1
                print(f"[Line {line_number}] Empty input/output, skipped.")
                continue

            records.append({"input": user_input, "output": output})

    return records, skipped


def write_jsonl(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            json.dump(record, outfile, ensure_ascii=False)
            outfile.write("\n")


def read_jsonl_records(input_path: Path):
    records = []

    with input_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[JSONL line {line_number}] Invalid JSON, skipped: {exc}")
                continue

            if not isinstance(record, dict):
                print(f"[JSONL line {line_number}] Not a JSON object, skipped.")
                continue

            user_input = record.get("input")
            output = record.get("output")

            if not isinstance(user_input, str) or not isinstance(output, str):
                print(f"[JSONL line {line_number}] Missing input/output fields, skipped.")
                continue

            user_input = user_input.strip()
            output = output.strip()

            if not user_input or not output:
                print(f"[JSONL line {line_number}] Empty input/output, skipped.")
                continue

            records.append({"input": user_input, "output": output})

    return records


def write_txt(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(f"{record['input']}|{record['output']}\n")


def convert():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    records, skipped = read_txt_records(INPUT_FILE)
    write_jsonl(records, JSONL_OUTPUT_FILE)
    rebuilt_records = read_jsonl_records(JSONL_OUTPUT_FILE)
    write_txt(rebuilt_records, TXT_OUTPUT_FILE)

    print()
    print("========== Finished ==========")
    print(f"Valid rows : {len(records)}")
    print(f"Skipped    : {skipped}")
    print(f"JSONL      : {JSONL_OUTPUT_FILE}")
    print(f"TXT        : {TXT_OUTPUT_FILE}")


if __name__ == "__main__":
    convert()
