import json
import random

NAMES = [
    "Gwendalyn", "Kaliah", "Avelyn", "Richmond", "Jak", "Kalayah", "Haya", "Ashlin",
    "Mostafa", "Scotty", "Jeancarlos", "Yasin", "Promise", "Lili", "Roberto", "Veronika",
    "Ceanna", "Aziyah", "Margaret", "Konrad", "Marcy", "Bastian", "Hilda", "Indigo",
    "Leiah", "Zeinab", "Oswaldo", "Arav", "Ruby", "Nayla", "Carlie", "Jameson", "Ianna",
    "Zayne", "Julian", "Neftali", "Ashby", "Amiyah", "Adara", "Mehki", "Jaydah", "Brynn",
    "Hadiya", "Skylynn", "Kenlee", "Marcello", "Caprice", "Desean", "Chimamanda", "Zaydin"
]

def filter_predictions(input_filename, output_filename, id_to_name, module):
    print(f"Reading {input_filename}...")
    with open(input_filename, 'r') as f:
        data = json.load(f)

    # Filter the data for only those student IDs and attach names
    filtered_data = []
    for item in data:
        if item.get('course_module') != module:
            continue
        student_id = item['student_id']
        if student_id in id_to_name:
            ordered_item = {}
            for k, v in item.items():
                ordered_item[k] = v
                if k == 'student_id':
                    # Insert name immediately after student_id
                    ordered_item['name'] = id_to_name[student_id]
            filtered_data.append(ordered_item)

    # Save to a new JSON file
    with open(output_filename, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Saved {len(filtered_data)} records for {len(id_to_name)} students to {output_filename}")

def get_unique_ids(filename, num_students, module):
    with open(filename, 'r') as f:
        data = json.load(f)
    seen_ids = set()
    unique_ids = []
    for item in data:
        if item.get('course_module') == module:
            student_id = item['student_id']
            if student_id not in seen_ids:
                seen_ids.add(student_id)
                unique_ids.append(student_id)
                if len(unique_ids) == num_students:
                    break
    return unique_ids

if __name__ == "__main__":
    num_students = len(NAMES)
    modules = ["AAA", "BBB", "DDD"]
    
    for module in modules:
        print(f"\nProcessing module {module}...")
        # Get unique IDs from one of the files
        unique_ids = get_unique_ids('rf_predictions.json', num_students, module)
        
        # Shuffle names and create a consistent mapping for both files
        shuffled_names = NAMES.copy()
        random.shuffle(shuffled_names)
        id_to_name = dict(zip(unique_ids, shuffled_names))
        
        # Filter rf_predictions.json
        filter_predictions('rf_predictions.json', f'rf_predictions_subset_{module}.json', id_to_name, module)
        
        # Filter lstm_predictions.json
        filter_predictions('lstm_predictions.json', f'lstm_predictions_subset_{module}.json', id_to_name, module)
