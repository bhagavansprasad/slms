import os
import glob

def extract_python_files(source_dir, output_file):
    """
    Combine all Python files into a single training text file
    """
    all_code = []
    
    # Find all .py files
    py_files = glob.glob(os.path.join(source_dir, "**/*.py"), recursive=True)
    
    print(f"Found {len(py_files)} Python files")
    
    for filepath in py_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Add file separator for context
                all_code.append(f"\n# File: {os.path.basename(filepath)}\n")
                all_code.append(content)
                all_code.append("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Combine all content
    combined_text = "".join(all_code)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print(f"✅ Combined {len(py_files)} files into {output_file}")
    print(f"   Total characters: {len(combined_text):,}")
    print(f"   Total lines: {combined_text.count(chr(10)):,}")
    
    return combined_text

if __name__ == "__main__":
    source_directory = "/home/bhagavan/my-git-repos/chromadb-basics"  # Where your .py files are
    output_file = "./data/training_data.txt"
    
    combined_text = extract_python_files(source_directory, output_file)
    
