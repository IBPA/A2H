import sys 
import requests
import csv
import datetime

def read_input(file_name):
    # read input
    # this should be a list of intervention names
    intervention_names = []
    with open(file_name, 'r', newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            intervention_names.append(row[0])
    # drop header
    intervention_names.pop(0)
    return intervention_names

def write_result(file_name, data, mode='w'):
    # create output
    with open(file_name, mode, newline='') as csvfile:
        fieldnames = ['original', 'found_word', 'cuid', 'type', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if mode == 'w':
            writer.writeheader()
        if data['cuid'] == None:
            writer.writerow(data)
        else:
            formatted_row = {'original': data['original'],
                            'found_word': data['found_word'],
                            'cuid': data['cuid']}
            if 'BN' in data:        
                for name in data['BN']:
                    writer.writerow({**formatted_row,'type':'BN', 'name': name})
            if 'IN' in data:        
                for name in data['IN']:
                    writer.writerow({**formatted_row,'type':'IN', 'name': name})

def fetch_rxcuid(intervention_name):
    # get rxcuid's
    rxcuid_search_url = 'https://rxnav.nlm.nih.gov/REST/rxcui.json?name='
    for word in intervention_name.split():
        r = requests.get(rxcuid_search_url + word)
        body = r.json()
        if 'rxnormId' in body['idGroup']:
            # grab the first, there will be many if search=1
            return word, body['idGroup']['rxnormId'][0]
    return None, None

def fetch_related_names(rxcuid):
    # get related names
    related_search_url = 'https://rxnav.nlm.nih.gov/REST/rxcui/{0}/related.json?tty=IN+BN'
    r = requests.get(related_search_url.format(rxcuid))
    body = r.json()
    concepts = {}
    for conceptGroup in body['relatedGroup']['conceptGroup']:
        group = conceptGroup['tty']
        if 'conceptProperties' in conceptGroup:
            concepts[group] = []
            for conceptProperty in conceptGroup['conceptProperties']:
                concepts[group].append(conceptProperty['name'].lower())
    return concepts

input_file_name = './interventions_raw.csv'
if len(sys.argv) == 2:
    input_file_name = sys.argv[1]

output_file_name = './interventions_processed.csv'
if len(sys.argv) == 3:
    output_file_name = sys.argv[2]

print(f"Reading input file: {input_file_name}")
print(f"Will create output file: {output_file_name}")

intervention_names = read_input(input_file_name)
print(f"Number of interventions to resolve: {len(intervention_names)}")

results = []
start = datetime.datetime.now()
for i, name in enumerate(intervention_names):
    if i%100 == 0 and i != 0 :
        rate = (datetime.datetime.now() - start).total_seconds() / i
        print(f"Processing intervention #{i}, estimated time remaining: {rate * (len(intervention_names) - i + 1)}s")
    elif i == 0:
        print(f"Processing intervention #{i}")
    
    found_word, cuid = fetch_rxcuid(name)
    concepts = fetch_related_names(cuid) if cuid != None else {}

    if i == 0:
        write_result(output_file_name, {'original': name, 'found_word': found_word, 'cuid': cuid, **concepts})
    else:
        write_result(output_file_name, {'original': name, 'found_word': found_word, 'cuid': cuid, **concepts}, 'a')