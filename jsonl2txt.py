import json

# Open the JSONL file for reading
with open('/workspace/data/dps_output/part-00000', 'r', encoding='utf-8') as jsonl_file:
    # Open a new text file for writing
    with open('corpus.txt', 'w', encoding='utf-8') as text_file:
        # Iterate through each line in the JSONL file
        for line in jsonl_file:
            # Load the JSON data from the line
            data = json.loads(line)
            
            # Extract the text field from the JSON data
            text = data['text']
            
            # Write the text to the text file
            text_file.write(text + '\n')

