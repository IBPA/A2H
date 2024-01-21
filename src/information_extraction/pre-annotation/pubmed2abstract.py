import sys 
import os

input_file_name = sys.argv[1]

output_dir = 'parsed_files'
if len(sys.argv) == 3:
    output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok = True)

z = []
with open(input_file_name, encoding='utf-8') as f:
    z = f.readlines()
    
new_file = []
while len(z) > 0:   
    current_line = z.pop(0)
    # last line check
    if not current_line.startswith('PMID'):
        new_file.append(current_line)
    else:     
        # rewind to the end of the abstract
        abstract = []
        while len(abstract) == 0:
            previous_line = new_file.pop()
            if previous_line == '\n':
                abstract.append(new_file.pop())
        
        # then the abstract will be preceeded by another \n
        abstract_start = False
        while not abstract_start:
            previous_line = new_file.pop()
            if previous_line != '\n' or (abstract[-1].lower().startswith('copyright') or abstract[-1][0].encode('utf-8') == u'\u00A9'.encode('utf-8')):
                abstract += [previous_line]
            else:
                abstract_start = True
        
        # strip off a copyright notice
        if abstract[0].lower().startswith('copyright') or abstract[0][0].encode('utf-8') == u'\u00A9'.encode('utf-8'):
            abstract.pop(0)
            if len(abstract) > 0 and abstract[0] == '\n':
                abstract.pop(0)
        
        PMID = current_line.split(' ')[1].strip('\n')
        with open(os.path.join(output_dir, PMID + '.txt'), 'w', encoding='utf-8') as f:
            for line in abstract[::-1]:
                f.write(line)