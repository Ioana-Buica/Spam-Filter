import argparse
import os
import json

def write_info(output_file):
    """
    Writes Project_name, Student_name, Alias_Student, Project_version in the output file.
    """
    info = {
        "Project_name": "SSOS",
        "Student_name": "Buica Ioana-Alexandra",
        "Alias_Student": "IO",
        "Project_version": "1"
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Information written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

def scan_folder(folder, output_file):
    """
    Scans a folder and writes a list of files and directories to the output file.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    try:
        file_structure = {
            "folder": folder,
            "contents": os.listdir(folder)
        }
        with open(output_file, 'w') as f:
            json.dump(file_structure, f, indent=4)
        print(f"Folder contents written to {output_file}.")
    except Exception as e:
        print(f"Error scanning folder or writing to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument("-info", metavar="output_file", type=str, help="Write output file name where to be written: Project_name, Student_name, Alias_Student, Project_version.")
    parser.add_argument("-scan", nargs=2, metavar=("folder", "output_file"), help="Scan the folder and filter the files.")

    args = parser.parse_args()

    if args.info:
        write_info(args.info)
    elif args.scan:
        scan_folder(args.scan[0], args.scan[1])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
