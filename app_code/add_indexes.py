import json
import os
import argparse
from typing import Optional


def add_indexes(input_file: str, output_file: Optional[str] = None) -> None:
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    ext = os.path.splitext(input_file)[1]

    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(input_file),
            f"{base_name}_indexes{ext}"
        )

    user_counter = 1
    assistant_counter = 1
    error_lines = []

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            if line_number % 1000 == 0:
                print(f"Przetworzono {line_number} linii...")

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Błąd parsowania JSON w linii {line_number}, pominięto.")
                error_lines.append(line_number)
                continue

            # Dodaj główny ID
            obj['id'] = f"{base_name}_{line_number}"

            # Przetwarzanie wiadomości
            for message in obj.get('messages', []):
                if message.get('role') == 'user':
                    message['id'] = f"query_{base_name}_{user_counter}"
                    user_counter += 1
                elif message.get('role') == 'assistant':
                    message['id'] = f"doc_{base_name}_{assistant_counter}"
                    assistant_counter += 1

            # Zapisz przetworzony obiekt jako nową linię JSON
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

    if error_lines:
        print("\n⚠️  JSON parsing errors occurred on the following lines:")
        print(", ".join(str(num) for num in error_lines))
    else:
        print("\n✅ Processing completed without errors.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign unique IDs to each input.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", required=False, help="Path to the output JSONL file (optional).")

    args = parser.parse_args()

    add_indexes(args.input_file, args.output_file)