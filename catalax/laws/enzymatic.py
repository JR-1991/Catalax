"""
Enzymatic Rate Law kinetic law functions.

This module contains kinetic law functions for the "Enzymatic Rate Law" category.
Functions are auto-generated from SBO kinetic law definitions.
"""

from sympy import sympify, Expr
from typing import Annotated

from ..model.utils import LOCALS


def michaelis_menten(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_m: Annotated[str, "Michaelis constant in quasi-steady state situation"] = "K_m",
) -> Expr:
    """First general rate equation for reactions involving enzymes, it was presented in "Victor Henri. Lois Generales de l'Action des Diastases. Paris, Hermann, 1903.". The reaction is assumed to be made of a reversible of the binding of the substrate to the enzyme, followed by the breakdown of the complex generating the product. Ten years after Henri, Michaelis and Menten presented a variant of his equation, based on the hypothesis that the dissociation rate of the substrate was much larger than the rate of the product generation. Leonor Michaelis, Maud Menten (1913). Die Kinetik der Invertinwirkung, Biochem. Z. 49:333-369.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        k_cat (str): Catalytic rate constant
        K_m (str): Michaelis constant in quasi-steady state situation

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(k_cat*E_t*S)/(K_m+S)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_m", locals=LOCALS)] = sympify(K_m, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def briggs_haldane(
    e_t: Annotated[str, "Concentration of enzyme"],
    s: Annotated[str, "Concentration of substrate"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_m: Annotated[str, "Michaelis constant in quasi-steady state situation"] = "K_m",
) -> Expr:
    """The Briggs-Haldane rate law is a general rate equation that does not require the restriction of equilibrium of Henri-Michaelis-Menten or irreversible reactions of Van Slyke, but instead make the hypothesis that the complex enzyme-substrate is in quasi-steady-state. Although of the same form than the Henri-Michaelis-Menten equation, it is semantically different since Km now represents a pseudo-equilibrium constant, and is equal to the ratio between the rate of consumption of the complex (sum of dissociation of substrate and generation of product) and the association rate of the enzyme and the substrate.

    Args:
        e_t (str): Concentration of enzyme
        s (str): Concentration of substrate
        k_cat (str): Catalytic rate constant
        K_m (str): Michaelis constant in quasi-steady state situation

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(k_cat*E_t*S)/(K_m+S)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_m", locals=LOCALS)] = sympify(K_m, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def normalised_enzymatic_for_unireactant_enzymes(
    s: Annotated[str, "Concentration of substrate"],
    k_cat: Annotated[str, "Catalytic rate constant"] = "k_cat",
    K_s: Annotated[str, "Michaelis constant"] = "K_s",
) -> Expr:
    """Kinetics of enzymes that react only with one substance, their substrate. The total enzyme concentration is considered to be equal to 1, therefore the maximal velocity equals the catalytic constant.

    Args:
        s (str): Concentration of substrate
        k_cat (str): Catalytic rate constant
        K_s (str): Michaelis constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(k_cat*S)/(K_s+S)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_non_modulated_non_interacting_bireactant_enzymes(
    e_t: Annotated[str, "Concentration of enzyme"],
    s_a: Annotated[str, "Concentration of substrate"],
    s_b: Annotated[str, "Concentration of substrate"],
    k_p: Annotated[str, "Catalytic rate constant"] = "k_p",
    K_1: Annotated[str, "Michaelis constant"] = "K_1",
    K_2: Annotated[str, "Michaelis constant"] = "K_2",
) -> Expr:
    """Kinetics of enzymes that react with two substances, their substrates, that bind independently. The enzymes do not catalyse the reactions in both directions.

    Args:
        e_t (str): Concentration of enzyme
        s_a (str): Concentration of substrate
        s_b (str): Concentration of substrate
        k_p (str): Catalytic rate constant
        K_1 (str): Michaelis constant
        K_2 (str): Michaelis constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(E_t*k_p*(S_a)/(K_1)*(S_b)/(K_2))/((1+(S_a)/(K_1))*(1+(S_b)/(K_2)))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("S_b", locals=LOCALS)] = sympify(s_b, locals=LOCALS)
    substitutions[sympify("k_p", locals=LOCALS)] = sympify(k_p, locals=LOCALS)
    substitutions[sympify("K_1", locals=LOCALS)] = sympify(K_1, locals=LOCALS)
    substitutions[sympify("K_2", locals=LOCALS)] = sympify(K_2, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_non_modulated_non_interacting_trireactant_enzymes(
    e_t: Annotated[str, "Concentration of enzyme"],
    s_a: Annotated[str, "Concentration of substrate"],
    s_b: Annotated[str, "Concentration of substrate"],
    s_3: Annotated[str, "Concentration of substrate"],
    k_p: Annotated[str, "Catalytic rate constant"] = "k_p",
    K_1: Annotated[str, "Michaelis constant"] = "K_1",
    K_2: Annotated[str, "Michaelis constant"] = "K_2",
    K_3: Annotated[str, "Michaelis constant"] = "K_3",
) -> Expr:
    """Kinetics of enzymes that react with three substances, their substrates, that bind independently. The enzymes do not catalyse the reactions in both directions.

    Args:
        e_t (str): Concentration of enzyme
        s_a (str): Concentration of substrate
        s_b (str): Concentration of substrate
        s_3 (str): Concentration of substrate
        k_p (str): Catalytic rate constant
        K_1 (str): Michaelis constant
        K_2 (str): Michaelis constant
        K_3 (str): Michaelis constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(E_t*k_p*(S_a)/(K_1)*(S_b)/(K_2)*(S_3)/(K_3))/((1+(S_a)/(K_1))*(1+(S_b)/(K_2))*(1+(S_3)/(K_3)))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("S_b", locals=LOCALS)] = sympify(s_b, locals=LOCALS)
    substitutions[sympify("S_3", locals=LOCALS)] = sympify(s_3, locals=LOCALS)
    substitutions[sympify("k_p", locals=LOCALS)] = sympify(k_p, locals=LOCALS)
    substitutions[sympify("K_1", locals=LOCALS)] = sympify(K_1, locals=LOCALS)
    substitutions[sympify("K_2", locals=LOCALS)] = sympify(K_2, locals=LOCALS)
    substitutions[sympify("K_3", locals=LOCALS)] = sympify(K_3, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def reversible_iso_uni_uni(
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    K_i: Annotated[str, "Isoinhibition constant"] = "K_i",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
) -> Expr:
    """Enzyme catalysed reaction involving one substrate and one product. Unlike the reversible uni-uni mechanism (SBO:0000326), the mechanism assumes an enzyme intermediate. Therefore, the free enzyme generated after the release of product from enzyme-product complex is not the same form as that which bind the substrate to form enzyme-substrate complex. Some permeases are thought to follow this mechanism, such that isomerization in the membrane may be accomplished through re-orientation in the membrane.

    Args:
        s (str): Concentration of substrate
        p (str): Concentration of product
        K_ms (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        K_i (str): Isoinhibition constant
        V_f (str): Forward maximal velocity
        K_eq (str): Equilibrium constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S-(P)/(K_eq)))/(S*(1+(P)/(K_i))+K_ms*(1+(P)/(K_mp)))", locals=LOCALS
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_i", locals=LOCALS)] = sympify(K_i, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def unmodulated_reversible_hill_type(
    s: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    K_s: Annotated[str, "Pseudo-dissociation constant for substrate"] = "K_s",
    K_p: Annotated[str, "Pseudo-dissociation constant for product"] = "K_p",
) -> Expr:
    """Reversible equivalent of Hill kinetics, where substrate and product bind co-operatively to the enzyme. A Hill coefficient (h) of greater than 1 indicates positive co-operativity between substrate and product, while h values below 1 indicate negative co-operativity.


    Args:
        s (str): Concentration of substrate
        p (str): Concentration of product
        K_eq (str): Equilibrium constant
        V_f (str): Forward maximal velocity
        K_s (str): Pseudo-dissociation constant for substrate
        K_p (str): Pseudo-dissociation constant for product

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S)/(K_s)*(1-(P)/(S*K_eq))*((S)/(K_s)+(P)/(K_p))^(h-1))/(1+((S)/(K_s)+(P)/(K_p))^h)",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("K_s", locals=LOCALS)] = sympify(K_s, locals=LOCALS)
    substitutions[sympify("K_p", locals=LOCALS)] = sympify(K_p, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def irreversible_michaelis_menten_for_two_substrates(
    a: Annotated[str, "Concentration of substrate"],
    b: Annotated[str, "Concentration of substrate"],
    e_t: Annotated[str, "Concentration of enzyme"],
    K_mA: Annotated[str, "Michaelis constant for substrate"] = "K_mA",
    K_mB: Annotated[str, "Michaelis constant for substrate"] = "K_mB",
    K_iA: Annotated[str, "Inhibitory constant"] = "K_iA",
    k_cat: Annotated[str, "Product catalytic rate constant"] = "k_cat",
) -> Expr:
    """Enzymatic rate law for an irreversible reaction involving two substrates and one product.

    Args:
        a (str): Concentration of substrate
        b (str): Concentration of substrate
        e_t (str): Concentration of enzyme
        K_mA (str): Michaelis constant for substrate
        K_mB (str): Michaelis constant for substrate
        K_iA (str): Inhibitory constant
        k_cat (str): Product catalytic rate constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify("(E_t*k_cat*A*B)/(K_iA*K_mB+K_mB*A+K_mA*B+A*B)", locals=LOCALS)

    substitutions = {}
    substitutions[sympify("A", locals=LOCALS)] = sympify(a, locals=LOCALS)
    substitutions[sympify("B", locals=LOCALS)] = sympify(b, locals=LOCALS)
    substitutions[sympify("E_t", locals=LOCALS)] = sympify(e_t, locals=LOCALS)
    substitutions[sympify("K_mA", locals=LOCALS)] = sympify(K_mA, locals=LOCALS)
    substitutions[sympify("K_mB", locals=LOCALS)] = sympify(K_mB, locals=LOCALS)
    substitutions[sympify("K_iA", locals=LOCALS)] = sympify(K_iA, locals=LOCALS)
    substitutions[sympify("k_cat", locals=LOCALS)] = sympify(k_cat, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def ordered_bi_bi_mechanism(
    s_a: Annotated[str, "Concentration of substrate"],
    s_b: Annotated[str, "Concentration of substrate"],
    p_p: Annotated[str, "Concentration of product"],
    p_q: Annotated[str, "Concentration of product"],
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
    K_ma: Annotated[str, "Michaelis constant for substrate"] = "K_ma",
    K_mb: Annotated[str, "Michaelis constant for substrate"] = "K_mb",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    K_mq: Annotated[str, "Michaelis constant for product"] = "K_mq",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_ib: Annotated[str, "Inhibitory constant"] = "K_ib",
    K_ip: Annotated[str, "Inhibitory constant"] = "K_ip",
) -> Expr:
    """Enzymatic rate law for a reaction involving two substrates and two products. The products P and then Q are released strictly in order, while the substrates are bound strictly in the order A and then B.

    Args:
        s_a (str): Concentration of substrate
        s_b (str): Concentration of substrate
        p_p (str): Concentration of product
        p_q (str): Concentration of product
        K_eq (str): Equilibrium constant
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity
        K_ma (str): Michaelis constant for substrate
        K_mb (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        K_mq (str): Michaelis constant for product
        K_ia (str): Inhibitory constant
        K_ib (str): Inhibitory constant
        K_ip (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*S_a*S_b-P_p*(P_q)/(K_eq))/(S_a*S_b*(1+(P_p)/(K_ip))+K_ma*S_b+K_mb*(S_a+K_ia)+(V_f)/(V_r*K_eq)*(K_mq*P_p*(1+(S_a)/(K_ia))+P_q*(K_mp*(1+(K_ma*S_b)/(K_ia*K_mb))+P_p*(1+(S_b)/(K_ib)))))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("S_b", locals=LOCALS)] = sympify(s_b, locals=LOCALS)
    substitutions[sympify("P_p", locals=LOCALS)] = sympify(p_p, locals=LOCALS)
    substitutions[sympify("P_q", locals=LOCALS)] = sympify(p_q, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)
    substitutions[sympify("K_ma", locals=LOCALS)] = sympify(K_ma, locals=LOCALS)
    substitutions[sympify("K_mb", locals=LOCALS)] = sympify(K_mb, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_mq", locals=LOCALS)] = sympify(K_mq, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_ib", locals=LOCALS)] = sympify(K_ib, locals=LOCALS)
    substitutions[sympify("K_ip", locals=LOCALS)] = sympify(K_ip, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def ordered_bi_uni_mechanism(
    s_a: Annotated[str, "Concentration of substrate"],
    s_b: Annotated[str, "Concentration of substrate"],
    p: Annotated[str, "Concentration of product"],
    K_ma: Annotated[str, "Michaelis constant for substrate"] = "K_ma",
    K_mb: Annotated[str, "Michaelis constant for substrate"] = "K_mb",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
) -> Expr:
    """Enzymatic rate for a reaction involving two substrates and one product. The substrates A and then B are bound strictly in order.

    Args:
        s_a (str): Concentration of substrate
        s_b (str): Concentration of substrate
        p (str): Concentration of product
        K_ma (str): Michaelis constant for substrate
        K_mb (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for product
        K_ia (str): Inhibitory constant
        K_eq (str): Equilibrium constant
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*S_a*S_b-(P)/(K_eq))/(S_a*S_b+K_ma*S_b+K_mb*S_a+(V_f)/(V_r*K_eq)*(K_mp+P*(1+(S_a)/(K_ia))))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("S_b", locals=LOCALS)] = sympify(s_b, locals=LOCALS)
    substitutions[sympify("P", locals=LOCALS)] = sympify(p, locals=LOCALS)
    substitutions[sympify("K_ma", locals=LOCALS)] = sympify(K_ma, locals=LOCALS)
    substitutions[sympify("K_mb", locals=LOCALS)] = sympify(K_mb, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def ordered_uni_bi_mechanism(
    s: Annotated[str, "Concentration of substrate"],
    p_p: Annotated[str, "Concentration of product"],
    p_q: Annotated[str, "Concentration of product"],
    K_ms: Annotated[str, "Michaelis constant for substrate"] = "K_ms",
    K_mq: Annotated[str, "Michaelis constant for product"] = "K_mq",
    K_mp: Annotated[str, "Michaelis constant for product"] = "K_mp",
    K_ip: Annotated[str, "Inhibitory constant"] = "K_ip",
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
) -> Expr:
    """Enzymatic rate law for a reaction with one substrate and two products. The products P and then Q are released in the strict order P and then Q.

    Args:
        s (str): Concentration of substrate
        p_p (str): Concentration of product
        p_q (str): Concentration of product
        K_ms (str): Michaelis constant for substrate
        K_mq (str): Michaelis constant for product
        K_mp (str): Michaelis constant for product
        K_ip (str): Inhibitory constant
        K_eq (str): Equilibrium constant
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S-(P_p*P_q)/(K_eq)))/(K_ms+S*(1+(P_p)/(K_ip))+(V_f)/(V_r*K_eq)*(K_mq*P_p+K_mp*P_q+P_p*P_q))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S", locals=LOCALS)] = sympify(s, locals=LOCALS)
    substitutions[sympify("P_p", locals=LOCALS)] = sympify(p_p, locals=LOCALS)
    substitutions[sympify("P_q", locals=LOCALS)] = sympify(p_q, locals=LOCALS)
    substitutions[sympify("K_ms", locals=LOCALS)] = sympify(K_ms, locals=LOCALS)
    substitutions[sympify("K_mq", locals=LOCALS)] = sympify(K_mq, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_ip", locals=LOCALS)] = sympify(K_ip, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation


def ping_pong_bi_bi_mechanism(
    s_a: Annotated[str, "Concentration of substrate"],
    s_b: Annotated[str, "Concentration of substrate"],
    p_p: Annotated[str, "Concentration of product"],
    p_q: Annotated[str, "Concentration of product"],
    K_eq: Annotated[str, "Equilibrium constant"] = "K_eq",
    V_f: Annotated[str, "Forward maximal velocity"] = "V_f",
    V_r: Annotated[str, "Reverse maximal velocity"] = "V_r",
    K_ma: Annotated[str, "Michaelis constant for substrate"] = "K_ma",
    K_mb: Annotated[str, "Michaelis constant for substrate"] = "K_mb",
    K_mp: Annotated[str, "Michaelis constant for substrate"] = "K_mp",
    K_mq: Annotated[str, "Michaelis constant for substrate"] = "K_mq",
    K_ia: Annotated[str, "Inhibitory constant"] = "K_ia",
    K_iq: Annotated[str, "Inhibitory constant"] = "K_iq",
) -> Expr:
    """Enzymatic rate law for a reaction involving two substrates and two products. The first product (P) is released after the first substrate (A) has been bound. The second product (Q) is released after the second substrate (B) has been bound.

    Args:
        s_a (str): Concentration of substrate
        s_b (str): Concentration of substrate
        p_p (str): Concentration of product
        p_q (str): Concentration of product
        K_eq (str): Equilibrium constant
        V_f (str): Forward maximal velocity
        V_r (str): Reverse maximal velocity
        K_ma (str): Michaelis constant for substrate
        K_mb (str): Michaelis constant for substrate
        K_mp (str): Michaelis constant for substrate
        K_mq (str): Michaelis constant for substrate
        K_ia (str): Inhibitory constant
        K_iq (str): Inhibitory constant

    Returns:
        str: The kinetic law equation as a string
    """
    equation = sympify(
        "(V_f*(S_a*S_b-(P_p*P_q)/(K_eq)))/(S_a*S_b+K_mb*S_a+K_ma*S_b*(1+(P_q)/(K_iq))+(V_f)/(V_r*K_eq)*(K_mq*P_p*(1+(S_a)/(K_ia))+P_q*(K_mp+P_p)))",
        locals=LOCALS,
    )

    substitutions = {}
    substitutions[sympify("S_a", locals=LOCALS)] = sympify(s_a, locals=LOCALS)
    substitutions[sympify("S_b", locals=LOCALS)] = sympify(s_b, locals=LOCALS)
    substitutions[sympify("P_p", locals=LOCALS)] = sympify(p_p, locals=LOCALS)
    substitutions[sympify("P_q", locals=LOCALS)] = sympify(p_q, locals=LOCALS)
    substitutions[sympify("K_eq", locals=LOCALS)] = sympify(K_eq, locals=LOCALS)
    substitutions[sympify("V_f", locals=LOCALS)] = sympify(V_f, locals=LOCALS)
    substitutions[sympify("V_r", locals=LOCALS)] = sympify(V_r, locals=LOCALS)
    substitutions[sympify("K_ma", locals=LOCALS)] = sympify(K_ma, locals=LOCALS)
    substitutions[sympify("K_mb", locals=LOCALS)] = sympify(K_mb, locals=LOCALS)
    substitutions[sympify("K_mp", locals=LOCALS)] = sympify(K_mp, locals=LOCALS)
    substitutions[sympify("K_mq", locals=LOCALS)] = sympify(K_mq, locals=LOCALS)
    substitutions[sympify("K_ia", locals=LOCALS)] = sympify(K_ia, locals=LOCALS)
    substitutions[sympify("K_iq", locals=LOCALS)] = sympify(K_iq, locals=LOCALS)

    for original, replacement in substitutions.items():
        equation = equation.subs(original, replacement)

    return equation
