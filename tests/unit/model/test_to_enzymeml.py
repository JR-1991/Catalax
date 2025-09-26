import pyenzyme as pe

from catalax.model.model import Model
from catalax.model.parameter import HDI


class TestToEnzymeML:
    def test_to_enzymeml_fresh(self):
        """Test conversion of a Catalax model to a fresh EnzymeML document.

        This test creates a complete Catalax model with states, reactions, ODEs,
        assignments, and parameters, then verifies that all components are correctly
        converted to the corresponding EnzymeML document structure.
        """
        # Create a new model
        model = Model(name="test")

        # Add states with different types
        model.add_states(
            S="Substrate",
            P="Product",
            E="Enzyme",
            ES="ES Complex",
        )

        # Set state types for proper EnzymeML categorization
        model.states["S"].type = "small_molecule"
        model.states["P"].type = "small_molecule"
        model.states["E"].type = "protein"
        model.states["ES"].type = "complex"

        # Add constants
        model.add_constants(E_total="Total Enzyme")

        # Add reactions with kinetic equations
        model.add_reaction(
            "S -> P",
            symbol="v_cat",
            equation="k_cat * E * S / (K_m + S)",
        )

        # Add ODEs for dynamic behavior
        model.add_odes(
            S="-v_cat",
            P="v_cat",
            E="-k_deg * E",
        )

        # Add assignments for derived quantities
        model.add_assignments(
            E_tot="E + ES",
        )

        # Assign parameter values and bounds with HDI information
        for param in model.parameters.values():
            param.value = 1.0
            param.lower_bound = 0.0
            param.upper_bound = 10.0
            param.hdi = HDI(
                lower_50=0.0,
                upper_50=10.0,
                lower=0.0,
                upper=10.0,
                q=0.5,
            )

        # Convert model to EnzymeML document
        doc = model.to_enzymeml()

        # Verify basic document properties
        assert doc.name == "test", "Document name should match model name"
        assert len(doc.small_molecules) > 0, "Document should contain small molecules"
        assert len(doc.proteins) > 0, "Document should contain proteins"
        assert len(doc.complexes) > 0, "Document should contain complexes"
        assert len(doc.parameters) > 0, "Document should contain parameters"
        assert len(doc.reactions) > 0, "Document should contain reactions"

        # Verify all states are correctly converted to appropriate species types
        for state in model.states.values():
            try:
                match state.type:
                    case "small_molecule":
                        small_molecule = doc.filter_small_molecules(name=state.name)[0]
                        assert (
                            small_molecule.id == str(state.symbol)
                        ), f"Small molecule ID should match state symbol for {state.name}"
                        assert (
                            small_molecule.name == state.name
                        ), f"Small molecule name should match state name for {state.name}"
                    case "protein":
                        protein = doc.filter_proteins(name=state.name)[0]
                        assert protein.id == str(
                            state.symbol
                        ), f"Protein ID should match state symbol for {state.name}"
                        assert (
                            protein.name == state.name
                        ), f"Protein name should match state name for {state.name}"
                    case "complex":
                        complex = doc.filter_complexes(name=state.name)[0]
                        assert complex.id == str(
                            state.symbol
                        ), f"Complex ID should match state symbol for {state.name}"
                        assert (
                            complex.name == state.name
                        ), f"Complex name should match state name for {state.name}"
            except ValueError:
                assert False, f"State {state.name} not found in EnzymeML document"

        # Verify all reactions are correctly converted
        for reaction in model.reactions.values():
            try:
                reaction_enzymeml = doc.filter_reactions(id=str(reaction.symbol))[0]

                assert reaction_enzymeml.id == str(
                    reaction.symbol
                ), f"Reaction ID should match reaction symbol for {reaction.symbol}"
                assert (
                    reaction_enzymeml.name == reaction.symbol
                ), f"Reaction name should match reaction symbol for {reaction.symbol}"

                # Verify kinetic law properties
                assert (
                    reaction_enzymeml.kinetic_law is not None
                ), f"Reaction {reaction.symbol} should have a kinetic law"
                assert reaction_enzymeml.kinetic_law.equation == str(
                    reaction.equation
                ), f"Kinetic law equation should match for reaction {reaction.symbol}"
                assert (
                    reaction_enzymeml.kinetic_law.equation_type
                    == pe.EquationType.RATE_LAW
                ), f"Kinetic law should be of type RATE_LAW for reaction {reaction.symbol}"

                # Verify reactants are correctly converted
                for reactant in reaction.reactants:
                    try:
                        reactant_enzymeml = reaction_enzymeml.filter_reactants(
                            species_id=str(reactant.state)
                        )[0]

                        assert (
                            reactant_enzymeml.species_id == str(reactant.state)
                        ), f"Reactant species ID should match state for {reactant.state}"
                    except ValueError:
                        assert (
                            False
                        ), f"Reactant {reactant.state} not found in EnzymeML document"

                    assert (
                        reactant_enzymeml.stoichiometry == reactant.stoichiometry
                    ), f"Reactant stoichiometry should match for {reactant.state}"

                # Verify products are correctly converted
                for product in reaction.products:
                    try:
                        product_enzymeml = reaction_enzymeml.filter_products(
                            species_id=str(product.state)
                        )[0]

                        assert product_enzymeml.species_id == str(
                            product.state
                        ), f"Product species ID should match state for {product.state}"
                    except ValueError:
                        assert (
                            False
                        ), f"Product {product.state} not found in EnzymeML document"

                    assert (
                        product_enzymeml.stoichiometry == product.stoichiometry
                    ), f"Product stoichiometry should match for {product.state}"
            except ValueError:
                assert (
                    False
                ), f"Reaction {reaction.symbol} not found in EnzymeML document"

        # Verify ODEs are correctly converted to equations
        for ode in model.odes.values():
            try:
                equation_enzymeml = doc.filter_equations(
                    species_id=str(ode.state.symbol)
                )[0]
                assert (
                    equation_enzymeml.species_id == str(ode.state.symbol)
                ), f"Equation species ID should match ODE state symbol for {ode.state.symbol}"
                assert equation_enzymeml.equation == str(
                    ode.equation
                ), f"Equation should match ODE equation for {ode.state.symbol}"
                assert (
                    equation_enzymeml.equation_type == pe.EquationType.ODE
                ), f"Equation type should be ODE for {ode.state.symbol}"
            except (ValueError, IndexError):
                assert (
                    False
                ), f"Equation {ode.state.symbol} not found in EnzymeML document"

        # Verify parameters are correctly converted with proper bounds
        for param in model.parameters.values():
            try:
                param_enzymeml = doc.filter_parameters(symbol=str(param.symbol))[0]
                assert param_enzymeml.id == str(
                    param.symbol
                ), f"Parameter ID should match parameter symbol for {param.symbol}"
                assert (
                    param_enzymeml.value == param.value
                ), f"Parameter value should match for {param.symbol}"
                assert (
                    param.hdi is not None
                ), f"Parameter HDI should not be None for {param.symbol}"
                assert (
                    param_enzymeml.lower_bound == param.hdi.lower_50
                ), f"Parameter lower bound should match HDI lower_50 for {param.symbol}"
                assert (
                    param_enzymeml.upper_bound == param.hdi.upper_50
                ), f"Parameter upper bound should match HDI upper_50 for {param.symbol}"
            except ValueError:
                assert False, f"Parameter {param.symbol} not found in EnzymeML document"

    def test_to_enzymeml_existing(self):
        """Test selective updating of an existing EnzymeML document with Catalax model data.

        This test verifies that when updating an existing EnzymeML document:
        - Existing custom names for species and reactions are preserved
        - Parameter names are preserved but numerical values are updated
        - Equations (kinetic laws, ODEs, assignments) are always updated
        - Missing components (parameters, equations) are added as needed
        """
        # Create a new model
        model = Model(name="test")

        # Add states with different types
        model.add_states(
            S="Substrate",
            P="Product",
            E="Enzyme",
            ES="ES Complex",
        )

        # Set state types for proper EnzymeML categorization
        model.states["S"].type = "small_molecule"
        model.states["P"].type = "small_molecule"
        model.states["E"].type = "protein"
        model.states["ES"].type = "complex"

        # Add constants
        model.add_constants(E_total="Total Enzyme")

        # Add reactions with kinetic equations
        model.add_reaction(
            "S -> P",
            symbol="v_cat",
            equation="k_cat * E * S / (K_m + S)",
        )

        # Add ODEs for dynamic behavior
        model.add_odes(
            S="-v_cat",
            P="v_cat",
            E="-k_deg * E",
        )

        # Add assignments for derived quantities
        model.add_assignments(
            E_tot="E + ES",
        )

        # Assign parameter values and bounds with HDI information
        for param in model.parameters.values():
            param.value = 1.0
            param.lower_bound = 0.0
            param.upper_bound = 10.0
            param.hdi = HDI(
                lower_50=0.0,
                upper_50=10.0,
                lower=0.0,
                upper=10.0,
                q=0.5,
            )

        # Create an existing EnzymeML document with pre-defined species and reactions
        existing_doc = pe.EnzymeMLDocument(name="existing")

        # Add species with specific names that should be preserved
        existing_doc.add_to_small_molecules(id="S", name="Custom Substrate Name")
        existing_doc.add_to_small_molecules(id="P", name="Custom Product Name")
        existing_doc.add_to_proteins(id="E", name="Custom Enzyme Name")
        existing_doc.add_to_complexes(id="ES", name="Custom Complex Name")

        # Add reactions with custom names (without kinetic laws initially)
        existing_doc.add_to_reactions(
            id="v_cat",
            name="Custom Catalytic Reaction",
            reactants=[pe.ReactionElement(species_id="S", stoichiometry=1.0)],
            products=[pe.ReactionElement(species_id="P", stoichiometry=1.0)],
            reversible=False,
        )

        # Add parameters with custom names but different values
        existing_doc.add_to_parameters(
            id="k_cat",
            name="Custom kcat Name",
            symbol="k_cat",
            value=0.5,  # Different from model value (1.0)
            lower_bound=0.1,  # Different from model bounds
            upper_bound=5.0,
        )

        # Convert model to update the existing EnzymeML document
        updated_doc = model.to_enzymeml(existing_doc)

        # Verify that custom names are preserved for species
        substrate = updated_doc.filter_small_molecules(id="S")[0]
        assert (
            substrate.name == "Custom Substrate Name"
        ), "Substrate custom name should be preserved"

        product = updated_doc.filter_small_molecules(id="P")[0]
        assert (
            product.name == "Custom Product Name"
        ), "Product custom name should be preserved"

        enzyme = updated_doc.filter_proteins(id="E")[0]
        assert (
            enzyme.name == "Custom Enzyme Name"
        ), "Enzyme custom name should be preserved"

        complex_species = updated_doc.filter_complexes(id="ES")[0]
        assert (
            complex_species.name == "Custom Complex Name"
        ), "Complex custom name should be preserved"

        # Verify that custom reaction names are preserved
        reaction = updated_doc.filter_reactions(id="v_cat")[0]
        assert (
            reaction.name == "Custom Catalytic Reaction"
        ), "Reaction custom name should be preserved"

        # Verify that kinetic laws are added/updated (equations should always be updated)
        assert (
            reaction.kinetic_law is not None
        ), "Kinetic law should be added to existing reaction"
        # The equation may be formatted differently due to symbolic math normalization
        expected_equation = str(model.reactions["v_cat"].equation)
        assert (
            reaction.kinetic_law.equation == expected_equation
        ), f"Kinetic law equation should be updated. Expected: {expected_equation}, Got: {reaction.kinetic_law.equation}"

        # Verify that parameter names are preserved but numerical values are updated
        param = updated_doc.filter_parameters(id="k_cat")[0]
        assert param.value == 1.0, "Parameter value should be updated from model"
        assert (
            param.lower_bound == 0.0
        ), "Parameter lower bound should be updated from model HDI"
        assert (
            param.upper_bound == 10.0
        ), "Parameter upper bound should be updated from model HDI"

        # Verify equations are always updated
        ode_equations = updated_doc.filter_equations(equation_type=pe.EquationType.ODE)
        assert len(ode_equations) > 0, "ODE equations should be added"

        s_equation = updated_doc.filter_equations(species_id="S")[0]
        expected_s_equation = str(model.odes["S"].equation)
        assert (
            s_equation.equation == expected_s_equation
        ), f"ODE equation should be updated. Expected: {expected_s_equation}, Got: {s_equation.equation}"

        # Verify assignment equations are also updated
        assignment_equations = updated_doc.filter_equations(
            equation_type=pe.EquationType.ASSIGNMENT
        )
        assert len(assignment_equations) > 0, "Assignment equations should be added"

        e_tot_equation = updated_doc.filter_equations(species_id="E_tot")[0]
        expected_assignment_equation = str(model.assignments["E_tot"].equation)
        assert (
            e_tot_equation.equation == expected_assignment_equation
        ), f"Assignment equation should be updated. Expected: {expected_assignment_equation}, Got: {e_tot_equation.equation}"
