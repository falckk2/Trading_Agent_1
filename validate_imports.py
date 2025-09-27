#!/usr/bin/env python3
"""
Comprehensive import and function call validation script.
Analyzes all Python files for import issues and missing function calls.
"""

import os
import ast
import sys
import importlib
import traceback
from pathlib import Path
from collections import defaultdict

def get_all_python_files(root_dir):
    """Get all Python files in the project."""
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]

        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files

def analyze_file_imports(file_path):
    """Analyze imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)
        imports = []

        class ImportCollector(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'name': alias.asname or alias.name,
                        'line': node.lineno
                    })
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'asname': alias.asname,
                        'line': node.lineno
                    })
                self.generic_visit(node)

        collector = ImportCollector()
        collector.visit(tree)
        return imports, None

    except SyntaxError as e:
        return [], f"Syntax error: {e}"
    except Exception as e:
        return [], f"Error reading file: {e}"

def check_import_validity(file_path, imports):
    """Check if imports are valid by attempting to import them."""
    issues = []

    # Set up path for relative imports
    file_dir = os.path.dirname(file_path)
    project_root = '/home/rehan/DeepAgent'

    # Add project root to Python path temporarily
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    for imp in imports:
        try:
            if imp['type'] == 'import':
                # Direct import
                importlib.import_module(imp['module'])
            else:
                # From import
                if imp['module'].startswith('.'):
                    # Relative import - calculate the actual module
                    rel_path = os.path.relpath(file_path, project_root)
                    package_parts = rel_path.replace('/', '.').replace('.py', '').split('.')

                    # Calculate relative module path
                    level = len(imp['module']) - len(imp['module'].lstrip('.'))
                    if level > 0:
                        base_package = '.'.join(package_parts[:-level])
                        if imp['module'][level:]:
                            full_module = f"{base_package}.{imp['module'][level:]}"
                        else:
                            full_module = base_package
                    else:
                        full_module = imp['module']
                else:
                    full_module = imp['module']

                if full_module:
                    try:
                        module = importlib.import_module(full_module)
                        if imp['name'] != '*' and not hasattr(module, imp['name']):
                            issues.append({
                                'type': 'missing_attribute',
                                'module': full_module,
                                'attribute': imp['name'],
                                'line': imp['line']
                            })
                    except ImportError:
                        issues.append({
                            'type': 'import_error',
                            'module': full_module,
                            'line': imp['line'],
                            'original_module': imp['module']
                        })

        except ImportError as e:
            issues.append({
                'type': 'import_error',
                'module': imp['module'],
                'line': imp['line'],
                'error': str(e)
            })
        except Exception as e:
            issues.append({
                'type': 'other_error',
                'module': imp['module'],
                'line': imp['line'],
                'error': str(e)
            })

    return issues

def analyze_function_calls(file_path):
    """Analyze function calls in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)
        calls = []

        class CallCollector(ast.NodeVisitor):
            def visit_Call(self, node):
                try:
                    if isinstance(node.func, ast.Name):
                        calls.append({
                            'type': 'function',
                            'name': node.func.id,
                            'line': node.lineno
                        })
                    elif isinstance(node.func, ast.Attribute):
                        # Method call
                        if hasattr(ast, 'unparse'):  # Python 3.9+
                            full_call = ast.unparse(node.func)
                        else:
                            # Fallback for older Python versions
                            full_call = 'unknown_method'
                        calls.append({
                            'type': 'method',
                            'name': full_call,
                            'line': node.lineno
                        })
                except:
                    # Skip calls we can't parse
                    pass

                self.generic_visit(node)

        collector = CallCollector()
        collector.visit(tree)
        return calls, None

    except Exception as e:
        return [], f"Error analyzing calls: {e}"

def main():
    project_root = '/home/rehan/DeepAgent'

    print("🔍 COMPREHENSIVE PROJECT VALIDATION")
    print("=" * 70)

    # Get all Python files
    py_files = get_all_python_files(f"{project_root}/crypto_trading")
    py_files = [f for f in py_files if 'test_ml_refactoring.py' not in f]  # Skip our test file

    print(f"📁 Found {len(py_files)} Python files to validate")

    # Track all issues
    all_import_issues = defaultdict(list)
    all_function_issues = defaultdict(list)
    files_with_syntax_errors = []

    print("\n📋 Phase 1: Import Validation")
    print("-" * 40)

    for i, py_file in enumerate(py_files):
        rel_path = os.path.relpath(py_file, project_root)

        # Analyze imports
        imports, import_error = analyze_file_imports(py_file)

        if import_error:
            files_with_syntax_errors.append((rel_path, import_error))
            continue

        # Check import validity
        if imports:
            import_issues = check_import_validity(py_file, imports)
            if import_issues:
                all_import_issues[rel_path] = import_issues

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(py_files)} files...")

    print(f"\n📋 Phase 2: Function Call Analysis")
    print("-" * 40)

    for i, py_file in enumerate(py_files):
        rel_path = os.path.relpath(py_file, project_root)

        if rel_path in [f[0] for f in files_with_syntax_errors]:
            continue  # Skip files with syntax errors

        # Analyze function calls
        calls, call_error = analyze_function_calls(py_file)

        if call_error:
            all_function_issues[rel_path].append({
                'type': 'analysis_error',
                'error': call_error
            })

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(py_files)} files...")

    # Report results
    print("\n" + "=" * 70)
    print("📊 VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n📁 Files analyzed: {len(py_files)}")
    print(f"❌ Syntax errors: {len(files_with_syntax_errors)}")
    print(f"⚠️  Import issues: {len(all_import_issues)}")
    print(f"🔧 Function issues: {len(all_function_issues)}")

    # Report syntax errors
    if files_with_syntax_errors:
        print(f"\n❌ FILES WITH SYNTAX ERRORS:")
        for file_path, error in files_with_syntax_errors:
            print(f"   {file_path}: {error}")

    # Report import issues
    if all_import_issues:
        print(f"\n⚠️  IMPORT ISSUES FOUND:")
        for file_path, issues in all_import_issues.items():
            print(f"\n   📄 {file_path}:")
            for issue in issues:
                if issue['type'] == 'import_error':
                    print(f"      Line {issue['line']}: Cannot import '{issue['module']}'")
                elif issue['type'] == 'missing_attribute':
                    print(f"      Line {issue['line']}: '{issue['attribute']}' not found in '{issue['module']}'")

    # Summary
    print(f"\n🎯 SUMMARY:")

    if not files_with_syntax_errors and not all_import_issues:
        print("✅ All files passed validation!")
        print("✅ No import errors found")
        print("✅ No syntax errors found")
        return True
    else:
        total_issues = len(files_with_syntax_errors) + len(all_import_issues)
        print(f"⚠️  Found {total_issues} files with issues")
        if files_with_syntax_errors:
            print(f"   - {len(files_with_syntax_errors)} files with syntax errors")
        if all_import_issues:
            print(f"   - {len(all_import_issues)} files with import issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)