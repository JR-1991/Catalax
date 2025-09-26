from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Set

import sympy as sp
from bigtree.node.node import Node
from sympy import Expr

from .equation import Equation
from .utils import LOCALS

if TYPE_CHECKING:
    from catalax.model import Model


class Assignment(Equation):
    symbol: str


def analyze_and_resolve_dependencies(
    model: Model,
) -> Dict[str, Expr]:
    """
    Analyze dependencies between assignments and resolve them by substitution.

    This is the main function that orchestrates the complete dependency analysis
    and resolution workflow. It handles both catalax models and custom assignment
    dictionaries, providing a comprehensive solution for symbolic substitution.

    The function performs the following steps:
    1. Extract assignments from input (catalax model or dictionary)
    2. Analyze dependencies using dual detection methods
    3. Build hierarchical dependency tree
    4. Determine correct evaluation order
    5. Perform symbol substitutions
    6. Display results if verbose mode enabled

    Args:
        model_or_assignments: Either a catalax Model object with an 'assignments'
                             attribute, or a dictionary mapping symbol names to
                             sympy expressions
        verbose: Whether to print detailed analysis information including tree
                structure, evaluation order, and substitution steps

    Returns:
        Dict[str, Any]: Comprehensive results dictionary containing:
            - 'original_assignments': Dict[str, sp.Expr] - Original symbol mappings
            - 'resolved_assignments': Dict[str, sp.Expr] - Fully expanded expressions
            - 'dependency_tree': Node - Root of the bigtree dependency structure
            - 'evaluation_order': List[str] - Symbols in correct evaluation order
            - 'assignment_dependencies': Dict[str, Set[str]] - Assignment-only dependencies
            - 'all_dependencies': Dict[str, Set[str]] - All dependencies (including external)

    Raises:
        ValueError: If input contains circular dependencies (though not explicitly checked)
        SympifyError: If string-based substitution produces invalid sympy expressions

    Note:
        The function uses a simplified tree-building approach for assignments with
        multiple dependencies. For complex dependency graphs, consider using a
        more sophisticated DAG-based approach.
    """
    # Step 1: Extract assignments from input
    assignments = _extract_assignments_from_input(model)

    # Step 2: Analyze dependencies
    assignment_dependencies = _analyze_dependencies(assignments)

    # Step 3: Build dependency tree
    dependency_tree = _build_dependency_tree(assignments, assignment_dependencies)

    # Step 4: Determine evaluation order
    evaluation_order = _determine_evaluation_order(dependency_tree)

    # Step 5: Perform substitutions
    resolved_assignments = _perform_substitutions(
        assignments,
        assignment_dependencies,
        evaluation_order,
    )

    return resolved_assignments


def _extract_assignments_from_input(
    model: Model,
) -> Dict[str, sp.Expr]:
    """
    Extract assignments dictionary from input (catalax model or dict).

    This function handles two types of input:
    1. A catalax Model object with an 'assignments' attribute
    2. A dictionary mapping symbol names to sympy expressions

    Args:
        model_or_assignments: Either a catalax Model object or a dictionary of
                             {symbol_name: sympy_expression} pairs

    Returns:
        Dict[str, sp.Expr]: Dictionary mapping symbol names to sympy expressions
    """

    assignments = {}
    for assignment in model.assignments.values():
        symbol = str(assignment.symbol)
        equation = assignment.equation
        assignments[symbol] = equation
    return assignments


def _analyze_dependencies(
    assignments: Dict[str, sp.Expr],
) -> Dict[str, Set[str]]:
    """
    Analyze dependencies between assignments using dual detection methods.

    This function identifies which assignments depend on other assignments using two approaches:
    1. Sympy free symbols analysis - detects symbols that appear as free variables
    2. String-based pattern matching - catches pre-substituted symbols in equation strings

    Args:
        assignments: Dictionary mapping symbol names to their sympy expressions

    Returns:
        - assignment_dependencies: Dict mapping each symbol to only assignment dependencies
    """
    assignment_dependencies: Dict[str, Set[str]] = {}

    for symbol, equation in assignments.items():
        # Method 1: Get all free symbols in the equation (dependencies)
        free_symbols = equation.free_symbols
        dep_symbols = {str(sym) for sym in free_symbols}

        # Method 2: Also check if assignment names appear in the string representation
        # (needed because some symbols may have been pre-substituted in the sympy expressions)
        equation_str = str(equation)
        string_deps: Set[str] = set()
        for other_symbol in assignments.keys():
            if other_symbol != symbol and other_symbol in equation_str:
                string_deps.add(other_symbol)

        # Combine both methods
        all_deps = dep_symbols | string_deps
        assignment_deps = all_deps.intersection(assignments.keys())

        assignment_dependencies[symbol] = assignment_deps

    return assignment_dependencies


def _build_dependency_tree(
    assignments: Dict[str, sp.Expr],
    assignment_dependencies: Dict[str, Set[str]],
) -> Node:
    """
    Build a hierarchical dependency tree using bigtree.

    Creates a tree structure where:
    - Root node contains assignments with no dependencies
    - Child nodes depend on their parent assignments
    - Tree depth represents dependency levels

    Args:
        assignments: Dictionary mapping symbol names to sympy expressions
        assignment_dependencies: Dictionary mapping symbols to their assignment dependencies

    Returns:
        Node: Root node of the dependency tree

    Note:
        For assignments with multiple dependencies, this implementation attaches to the
        first dependency found. In complex cases, you might want a more sophisticated
        approach like creating a DAG (Directed Acyclic Graph).

    Example:
        >>> assignments = {'a': x, 'b': y*a, 'c': b + z}
        >>> assignment_deps = {'a': set(), 'b': {'a'}, 'c': {'b'}}
        >>> tree = _build_dependency_tree(assignments, assignment_deps)
        >>> # Tree structure: Root -> a -> b -> c
    """
    nodes: Dict[str, Node] = {}
    root = Node("Root")

    # Create all nodes
    for symbol in assignments.keys():
        nodes[symbol] = Node(symbol)

    # Build dependency relationships
    for symbol, assignment_deps in assignment_dependencies.items():
        if not assignment_deps:
            # No dependencies on other assignments - attach to root
            nodes[symbol].parent = root
        else:
            # Attach to the first dependency (simplified approach)
            primary_dep = next(iter(assignment_deps))
            nodes[symbol].parent = nodes[primary_dep]

    return root


def _determine_evaluation_order(root: Node) -> List[str]:
    """
    Determine the correct evaluation order for symbol substitutions.

    The evaluation order is determined by tree depth, ensuring that:
    1. Independent assignments (no dependencies) are evaluated first
    2. Dependent assignments are evaluated after their dependencies
    3. This guarantees that when substituting, all required symbols are already resolved

    Args:
        root: Root node of the dependency tree

    Returns:
        List[str]: Ordered list of symbol names for evaluation (least dependent first)

    Example:
        >>> # For tree: Root -> E_denom -> E -> [dPG, dCEX]
        >>> order = _determine_evaluation_order(root)
        >>> print(order)
        ['E_denom', 'E', 'dPG', 'dCEX']
    """
    all_descendants = list(root.descendants)
    sorted_nodes = sorted(all_descendants, key=lambda x: x.depth)
    return [node.node_name for node in sorted_nodes if node.node_name != "Root"]


def _perform_substitutions(
    assignments: Dict[str, sp.Expr],
    assignment_dependencies: Dict[str, Set[str]],
    evaluation_order: List[str],
    verbose: bool = False,
) -> Dict[str, sp.Expr]:
    """
    Perform symbol substitutions to resolve all dependencies.

    This function iterates through assignments in dependency order and substitutes
    dependent symbols with their resolved expressions. Uses dual substitution methods:
    1. Direct sympy substitution for symbols in free_symbols
    2. String-based substitution for pre-expanded symbols

    Args:
        assignments: Dictionary mapping symbol names to sympy expressions
        assignment_dependencies: Dictionary mapping symbols to their assignment dependencies
        evaluation_order: List of symbols in correct evaluation order
        verbose: Whether to print detailed substitution information

    Returns:
        Dict[str, sp.Expr]: Dictionary with fully resolved assignments where all
                           assignment dependencies have been substituted

    Example:
        >>> # Given: z = a+b, y = x*z
        >>> assignments = {'z': a+b, 'y': x*z}
        >>> assignment_deps = {'z': set(), 'y': {'z'}}
        >>> order = ['z', 'y']
        >>> resolved = _perform_substitutions(assignments, assignment_deps, order)
        >>> print(resolved['y'])  # x*(a + b)

    Note:
        The function modifies expressions in-place based on evaluation order,
        ensuring that when a symbol is substituted, all its dependencies
        have already been resolved.
    """
    resolved_assignments = copy.deepcopy(assignments)

    if verbose:
        print("=== Performing Substitutions ===")

    for symbol in evaluation_order:
        deps = assignment_dependencies[symbol]
        if deps:
            original_eq = resolved_assignments[symbol]
            substituted_eq = original_eq

            for dep_symbol in deps:
                # Get the resolved equation for the dependency
                dep_equation = resolved_assignments[dep_symbol]

                # Try multiple approaches to find and substitute the symbol
                # 1. Direct sympy symbol substitution
                dep_sympy_symbol = sp.Symbol(dep_symbol)
                if dep_sympy_symbol in substituted_eq.free_symbols:
                    substituted_eq = substituted_eq.subs(dep_sympy_symbol, dep_equation)
                else:
                    # 2. String-based substitution and re-parsing
                    # This handles cases where the symbol appears in the string but not as a free symbol
                    eq_str = str(substituted_eq)
                    if dep_symbol in eq_str:
                        # Replace the string and re-parse
                        new_eq_str = eq_str.replace(dep_symbol, f"({dep_equation})")
                        try:
                            substituted_eq = sp.sympify(new_eq_str, locals=LOCALS)
                        except (sp.SympifyError, ValueError, TypeError):
                            # If sympify fails, keep the original
                            if verbose:
                                print(
                                    f"  Warning: Could not parse substituted equation for {dep_symbol}"
                                )
                            pass

            resolved_assignments[symbol] = substituted_eq  # type: ignore

            if verbose:
                print(f"{symbol}:")
                print(f"  Original: {original_eq}")
                print(f"  Resolved: {substituted_eq}")
                print()

    return resolved_assignments
