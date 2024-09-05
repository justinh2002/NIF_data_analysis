import re


with open('Turb.log', 'r') as file:
    lines = file.readlines()


with open('extracted_data.txt', 'w') as output_file:
    
    
    for line in lines:
        
        
        match = re.search(r"step: n=(\d+) t=(\S+)", line)
        
        if match:
            # Extract step and time values
            step = match.group(1)
            time = match.group(2)
            
            
            output_file.write(f"{time}\n")
