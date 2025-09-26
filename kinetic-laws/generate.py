#!/usr/bin/env python3
"""
Script to generate Python files with kinetic law functions from sbo_laws.json
Groups functions by category and creates separate files in catalax/laws/

Usage:
    python kinetic-laws/generate.py

This script should be run from the project root directory.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from jinja2 import Template

DEFAULT_PREAMBLE = '''"""
{category} kinetic law functions.

This module contains kinetic law functions for the "{category}" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS

'''


def check_root_directory():
    """Check if we're in the project root directory."""
    current_dir = Path.cwd()
    catalax_dir = current_dir / "catalax"
    kinetic_laws_dir = current_dir / "kinetic-laws"

    if not catalax_dir.exists() or not catalax_dir.is_dir():
        print("Error: 'catalax' directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    if not kinetic_laws_dir.exists() or not kinetic_laws_dir.is_dir():
        print("Error: 'kinetic-laws' directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)

    return current_dir


def sanitize_filename(category):
    """Convert category name to valid Python filename"""
    # Remove "Rate Law" and sanitize
    filename = category.replace("Rate Law", "").strip()
    # Convert to snake_case
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.replace("-", "_")
    filename = filename.replace("(", "")
    filename = filename.replace(")", "")
    filename = filename.replace(",", "")
    # Remove any double underscores
    while "__" in filename:
        filename = filename.replace("__", "_")
    # Remove leading/trailing underscores
    filename = filename.strip("_")
    return filename


def sanitize_function_name(name):
    """Convert law name to valid Python function name"""
    # Remove "rate law" from the name first
    func_name = name.lower()
    func_name = func_name.replace("rate law", "").strip()
    func_name = func_name.replace(" ", "_")
    func_name = func_name.replace(",", "")
    func_name = func_name.replace("-", "_")
    func_name = func_name.replace("(", "")
    func_name = func_name.replace(")", "")
    func_name = func_name.replace("'", "")
    func_name = func_name.replace("/", "_")
    # Remove any double underscores
    while "__" in func_name:
        func_name = func_name.replace("__", "_")
    # Remove leading/trailing underscores
    func_name = func_name.strip("_")
    return func_name


def main():
    """Main function to generate kinetic law Python files."""
    # Check if we're in the correct directory
    root_dir = check_root_directory()

    # Define paths relative to root
    json_file = root_dir / "kinetic-laws" / "sbo_laws.json"
    template_file = root_dir / "kinetic-laws" / "law_function.jinja2"
    laws_dir = root_dir / "catalax" / "laws"

    # Load the JSON data
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            laws_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)

    # Load the Jinja2 template
    try:
        with open(template_file, "r", encoding="utf-8") as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {template_file}")
        sys.exit(1)

    template = Template(template_content)

    # Group laws by category
    categories = defaultdict(list)
    for law in laws_data:
        category = law["category"]
        categories[category].append(law)

    # Create the laws directory if it doesn't exist
    laws_dir.mkdir(exist_ok=True)

    print(f"Found {len(categories)} categories:")
    for category in sorted(categories.keys()):
        print(f"  - {category} ({len(categories[category])} laws)")

    # Generate Python files for each category
    for category, laws in categories.items():
        filename = sanitize_filename(category)
        filepath = laws_dir / f"{filename}.py"

        print(f"\nGenerating {filepath}...")

        # Start with imports and module docstring
        file_content = DEFAULT_PREAMBLE.format(category=category)

        # Generate functions for each law in this category
        generated_count = 0
        for law in laws:
            try:
                # Add the sanitized function name to the law data for template
                law_with_func_name = law.copy()
                law_with_func_name["function_name"] = sanitize_function_name(
                    law["name"]
                )

                function_code = template.render(**law_with_func_name)
                file_content += function_code + "\n\n"
                generated_count += 1
            except Exception as e:
                print(
                    f"  Warning: Failed to generate function for '{law['name']}': {e}"
                )
                continue

        # Write the file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file_content)
        except IOError as e:
            print(f"Error: Could not write to {filepath}: {e}")
            continue

        print(f"  Generated {generated_count} functions")

    # Update the __init__.py file to import all modules
    init_file = laws_dir / "__init__.py"
    try:
        with open(init_file, "w", encoding="utf-8") as f:
            f.write('"""Kinetic law functions organized by category."""\n\n')
            for category in sorted(categories.keys()):
                module_name = sanitize_filename(category)
                f.write(f"from . import {module_name}\n")

            f.write("\n\n__all__ = [\n")
            for category in sorted(categories.keys()):
                module_name = sanitize_filename(category)
                f.write(f'    "{module_name}",\n')
            f.write("]\n")
    except IOError as e:
        print(f"Error: Could not write to {init_file}: {e}")
        sys.exit(1)

    print(
        f"\nGeneration complete! Created {len(categories)} Python files in {laws_dir}"
    )


if __name__ == "__main__":
    main()
