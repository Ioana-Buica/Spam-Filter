import argparse
import os
import json
import re
from collections import defaultdict, Counter
from math import log


def write_info(output_file):
    """
    Writes Project_name, Student_name, Alias_Student, Project_version in the output file.
    """
    try:
        with open(output_file, "w") as f:
            
            Project_name = "SSOS"
            Student_name = "Buica Ioana-Alexandra"
            Alias_Student = "IO"
            Project_version = "1"

            f.write(
                f"Project name: {Project_name}\nStudent name: {Student_name}\nAlias Student: {Alias_Student}\nProject version: {Project_version}\n"
            )
            
        print(f"Information written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def scan_folder(folder, output_file):
    """
    Scans a folder and filter the files.
    """
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    try:
        with open(output_file, "w") as file:
            # Scan the folder (not recursively) 
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    try:
                        # Read de info from the file
                        with open(item_path, "r") as f:
                            data = f.readlines()

                            # Extract subject and content
                            if data:
                                subject = data[0].strip()  # First line as subject
                                content = "".join(
                                    data[1:]
                                ).strip()  # Remaining lines as content

                                # category = predict(subject, content)

                        f.close()
                    except Exception as e:
                        print(f"Error processing file '{item}': {e}")

                    category = "cln"
                    file.write(f"{item}|{category}\n")
        file.close()
        print(f"Information written to {output_file}.")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Anti-spam filter.")
    parser.add_argument(
        "-info",
        metavar="output_file",
        type=str,
        help="Write output file name where to be written: Project_name, Student_name, Alias_Student, Project_version.",
    )
    parser.add_argument(
        "-scan",
        nargs=2,
        metavar=("folder", "output_file"),
        help="Scan the folder and filter the files.",
    )

    args = parser.parse_args()

    if args.info:
        write_info(args.info)
    elif args.scan:
        scan_folder(args.scan[0], args.scan[1])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
