"""
Inhibition kinetic law functions.

This module contains kinetic law functions for the "Inhibition" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS


def competitive_inhibition_of_irreversible_unireactant_enzymes_by_two_exclusive_inhibitors(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    i_a: Annotated[str, "Concentration of inhibitor"],
    i_b: Annotated[str, "Concentration of inhibitor"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_ib: Annotated[str, "Inhibitory constant"] = "K_ib",
) -> Expr:
    """Inhibition of a unireactant enzyme by two inhibitors that bind to the free enzyme on the same binding site than the substrate.  The enzymes do not catalyse the reactions in both directions.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        i_a (str): Concentration of inhibitor
        i_b (str): Concentration of inhibitor
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_ia (str): Inhibitory constant
        K_ib (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(k_cat*E_t*S)/(K_s*(1+(I_a)/(K_ia)+(I_b)/(K_ib))+S)", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("I_a", locals=LOCALS)] = sympify(i_a, locals=LOCALS)
    substitutions[sympify("I_b", locals=LOCALS)] = sympify(i_b, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_ib", locals=LOCALS)] = sympify(K_ib, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def inhibition_of_irreversible_unireactant_enzymes_by_single_competing_substrate_with_product_inhibition(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    s_a: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    p_a: Annotated[str, "Concentration of product"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_sa: Annotated[str, "Michaelis constant"] = "K_sa",
    K_p: Annotated[str, "Inhibitory constant"] = "K_p",
    K_pa: Annotated[str, "Inhibitory constant"] = "K_pa",
) -> Expr:
    """Inhibition of a unireactant enzyme by a competing substrate (Sa) that binds to the free enzyme on the same binding site, and competitive inhibition by a product (P) and an alternative product (Pa). The enzyme does not catalyse the reactions in both directions.


    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        s_a (str): Concentration of substrate
        p (str): Concentration of product
        p_a (str): Concentration of product
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_sa (str): Michaelis constant
        K_p (str): Inhibitory constant
        K_pa (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(k_cat*E_t*S)/(K_s*(1+(S_a)/(K_sa)+(P)/(K_pa)+(P_a)/(K_pa))+S)", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("P_a", locals=LOCALS)] = sympify(p_a, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_sa", locals=LOCALS)] = sympify(K_sa, locals=LOCALS)
    substitutions[sympify("K_p", locals=LOCALS)] = sympify(K_p, locals=LOCALS)
    substitutions[sympify("K_pa", locals=LOCALS)] = sympify(K_pa, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def competitive_inhibition_of_irreversible_unireactant_enzyme_by_product(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_p: Annotated[str, "Inhibitory constant"] = "K_p",
) -> Expr:
    """Inhibition of a unireactant enzyme by a competing product (P) that binds to the free enzyme on the same binding site. The enzyme does not catalyse the reactions in both directions.


    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        p (str): Concentration of product
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_p (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(k_cat*E_t*S)/(K_s*(1+(P)/(K_p))+S)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_p", locals=LOCALS)] = sympify(K_p, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def simple_competitive_inhibition_of_irreversible_unireactant_enzymes_by_two_non_exclusive_non_cooperative_inhibitors(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    i_a: Annotated[str, "Concentration of inhibitor"],
    i_b: Annotated[str, "Concentration of inhibitor"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_ib: Annotated[str, "Inhibitory constant"] = "K_ib",
) -> Expr:
    """Inhibition of a unireactant enzyme by two inhibitors that can bind independently once to the free enzyme and preclude the binding of the substrate. The enzymes do not catalyse the reactions in both directions.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        i_a (str): Concentration of inhibitor
        i_b (str): Concentration of inhibitor
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_ia (str): Inhibitory constant
        K_ib (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(k_cat*E_t*S)/(K_s*(1+(I_a)/(K_ia)+(I_b)/(K_ib)+(I_a*I_b)/(K_ia*K_ib))+S)",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("I_a", locals=LOCALS)] = sympify(i_a, locals=LOCALS)
    substitutions[sympify("I_b", locals=LOCALS)] = sympify(i_b, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_ib", locals=LOCALS)] = sympify(K_ib, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def simple_irreversible_non_competitive_inhibition_of_unireactant_enzymes(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    i: Annotated[str, "Concentration of inhibitor"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_i: Annotated[str, "Inhibitory constant"] = "K_i",
) -> Expr:
    """Inhibition of a unireactant enzyme by one inhibitor that can bind to the complex enzyme-substrate and the free enzyme with the same equilibrium constant, and totally prevent the catalysis.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        i (str): Concentration of inhibitor
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_i (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(k_cat*E_t*S)/(S*(1+(I)/(K_i))+K_s*(1+(I)/(K_i)))", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("I", locals=LOCALS)] = sympify(i, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def non_competitive_inhibition_of_irreversible_unireactant_enzymes_by_two_exclusively_binding_inhibitors(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    i_a: Annotated[str, "Concentration of inhibitor"],
    i_b: Annotated[str, "Concentration of inhibitor"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_ib: Annotated[str, "Inhibitory constant"] = "K_ib",
) -> Expr:
    """Inhibition of unireactant enzymes by two inhibitors that can bind to the complex enzyme-substrate and the free enzyme with the same equilibrium constant and totally prevent the catalysis.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        i_a (str): Concentration of inhibitor
        i_b (str): Concentration of inhibitor
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant
        K_ia (str): Inhibitory constant
        K_ib (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(k_cat*E_t*S)/(S*(1+(I_a)/(K_ia)+(I_b)/(K_ib))+K_s*(1+(I_a)/(K_ia)+(I_b)/(K_ib)))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("I_a", locals=LOCALS)] = sympify(i_a, locals=LOCALS)
    substitutions[sympify("I_b", locals=LOCALS)] = sympify(i_b, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_ib", locals=LOCALS)] = sympify(K_ib, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def mixed_type_inhibition_of_reversible_enzymes_by_mutually_exclusive_inhibitors(
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    i: Annotated[str, "Concentration of inhibitor"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
    K_is: Annotated[str, "Inhibitory constant"] = "K_is",
    K_ic: Annotated[str, "Inhibitory constant"] = "K_ic",
) -> Expr:
    """Reversible inhibition of a unireactant enzyme by inhibitors that can bind to the enzyme-substrate complex and to the free enzyme with the same equilibrium constant. The inhibitor is noncompetitive with the substrate.

    Args:
        s (str): Concentration of substrate
        p (str): Concentration of product
        i (str): Concentration of inhibitor
        K_ms (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity
        K_is (str): Inhibitory constant
        K_ic (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S)/(K_ms)-V_r*(P)/(K_mp))/(1+(I)/(K_is)+((S)/(K_ms)+(P)/(K_mp))*(1+(I)/(K_ic)))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("I", locals=LOCALS)] = sympify(i, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)
    substitutions[sympify("K_is", locals=LOCALS)] = sympify(K_is, locals=LOCALS)
    substitutions[sympify("K_ic", locals=LOCALS)] = sympify(K_ic, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_allosteric_inhibition(
    s: Annotated[str, "Concentration of substrate"],
    i: Annotated[str, "Concentration of inhibitor"],
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_s: Annotated[str, "Dissociation constant"] = "K_s",
    L: Annotated[str, "Equilibrium constant"] = "L",
    K_i: Annotated[str, "Inhibitory constant"] = "K_i",
) -> Expr:
    """Enzymatic rate law which follows from the allosteric concerted model (symmetry model or MWC model).This states that enzyme subunits can assume one of two conformational states (relaxed or tense), and that the state of one subunit is shared or enforced on the others. The binding of a ligand to a site other than that bound by the substrate (active site) can shift the conformation from one state to the other. L represents the equilibrium constant between active and inactive states of the enzyme, and n represents the number of binding sites for the substrate and inhibitor.

    Args:
        s (str): Concentration of substrate
        i (str): Concentration of inhibitor
        V (str): Forward maximal velocity
        K_s (str): Dissociation constant
        L (str): Equilibrium constant
        K_i (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V*S*(K_s+S)^(n-1))/(L*(K_s*(1+(I)/(K_i)))^n+(K_s+S)^n)", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("I", locals=LOCALS)] = sympify(i, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("L", locals=LOCALS)] = sympify(L, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def reversible_competitive_inhibition_by_one_inhibitor(
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    i: Annotated[str, "Concentration of inhibitor"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
    K_i: Annotated[str, "Inhibitory constant"] = "K_i",
) -> Expr:
    """This enzymatic rate law involves one substrate, one product and one modifier. The modifier acts as a competitive inhibitor with the substrate at the enzyme binding site; The modifier (inhibitor) reversibly bound to the enzyme blocks access to the substrate. The inhibitor has the effect of increasing the apparent Km.

    Args:
        s (str): Concentration of substrate
        p (str): Concentration of product
        i (str): Concentration of inhibitor
        K_ms (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity
        K_i (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S)/(K_ms)-V_r*(P)/(K_mp))/(1+(S)/(K_ms)+(P)/(K_mp)+(I)/(K_i))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("I", locals=LOCALS)] = sympify(i, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def reversible_empirical_allosteric_inhibition_by_one_inhibitor(
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    i: Annotated[str, "Concentration of inhibitor"],
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    K_i: Annotated[str, "Inhibitory constant"] = "K_i",
) -> Expr:
    """Enzymatic rate law where the reversible binding of one ligand decreases the affinity for substrate at other active sites. The ligand does not bind the same site as the substrate on the enzyme. This is an empirical equation, where n represents the Hill coefficient.

    Args:
        s (str): Concentration of substrate
        p (str): Concentration of product
        i (str): Concentration of inhibitor
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity
        K_ms (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        K_i (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S)/(K_ms)-V_r*(P)/(K_mp))/(1+(S)/(K_ms)+(P)/(K_mp)+((I)/(K_i))^n)",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("I", locals=LOCALS)] = sympify(i, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_substrate_inhibition(
    s: Annotated[str, "Concentration of substrate"],
    K_m: Annotated[str, "Michaelis constant"] = "K_m",
    V: Annotated[str, "Forward maximal velocity"] = "V",
    K_i: Annotated[str, "Inhibitory constant"] = "K_i",
) -> Expr:
    """Enzymatic rate law where the substrate for an enzyme also acts as an irreversible inhibitor. This may entail a second (non-active) binding site for the enzyme. The inhibition constant is then the dissociation constant for the substrate from this second site.


    Args:
        s (str): Concentration of substrate
        K_m (str): Michaelis constant
        V (str): Forward maximal velocity
        K_i (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(V*S)/(K_m+S+K_m*((S)/(K_i))^2)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("K_m", locals=LOCALS)] = sympify(K_m, locals=LOCALS)
    substitutions[sympify("V", locals=LOCALS)] = sympify(V, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation
