"""Benchmark data for embed_model_compare.py: a fixed corpus of technical
documents and two ground-truth query tracks.

Structure:
    CORPUS           — (doc_id, title, text); one chunk per doc (chunking is
                       not under test, the embedding model is)
    SEMANTIC_QUERIES — (query, expected_doc_id, is_trap); paraphrase-style
                       questions — no lexical overlap with the answer by
                       design; the dense/embedding track
    LEXICAL_QUERIES  — (query, expected_doc_id); short keyword queries where
                       every token literally appears in the target doc — the
                       BM25-fair track

Splitting the tracks makes both retrieval branches measurable: BM25 should
win the lexical track and return nothing on the semantic one (the library's
FTS5 branch ANDs every token); embedding models show the reverse. The
hybrid pipeline exists to get both right at once.

`is_trap` marks semantic queries whose vocabulary points at a *different*
doc than the expected one — they differentiate models that match keywords
from models that match meaning.
"""

from __future__ import annotations

CORPUS = [
    # -- home energy --------------------------------------------------------
    (
        "solar",
        "Rooftop solar PV",
        "A rooftop photovoltaic system converts sunlight into direct current, "
        "which an inverter turns into grid-compatible alternating current. A "
        "typical 4 kWp array on a south-facing roof in a temperate climate "
        "yields roughly 3,500 kWh per year. Upfront cost is dominated by panels "
        "and installation labour; payback depends on how much generation you "
        "self-consume rather than export, since exported units earn far less "
        "than imported units cost. Most systems are grid-tied without "
        "batteries and drop output sharply under cloud or shading.",
    ),
    (
        "wind",
        "Domestic wind turbine",
        "Small wind turbines for homes range from 1 to 10 kW. Output is "
        "extremely site-sensitive: urban rooftops see turbulent, slow air and "
        "often produce almost nothing, while an exposed rural pole mount can "
        "be worthwhile. Planning permission is usually required because of "
        "visual impact, and neighbours sometimes object to blade noise and "
        "flicker. Maintenance means inspecting the mast, guy ropes and inverter "
        "every few years. In most home settings wind complements solar rather "
        "than replacing it, since winter wind and summer sun peak in different "
        "seasons.",
    ),
    (
        "insulation",
        "Home insulation",
        "Heat lost through the building fabric drives most heating bills. "
        "Loft insulation is the cheapest measure: 300 mm of mineral wool "
        "dramatically cuts the share lost through the roof. Cavity wall "
        "filling is next; solid walls need internal or external boards, which "
        "cost far more. Performance is compared with U-values, where lower is "
        "better. Draught-proofing around doors, windows and loft hatches is "
        "nearly free and improves comfort immediately. Insulation should "
        "generally precede any heating-system upgrade, because a fabric-first "
        "approach lets a smaller, cheaper system do the same job.",
    ),
    (
        "heatpump",
        "Air-source heat pump",
        "An air-source heat pump moves heat from outside air into water for "
        "radiators or underfloor heating instead of burning fuel to make heat. "
        "Efficiency is the coefficient of performance: 3.5 means 3.5 kWh of "
        "warmth per kWh of electricity. Output drops as outdoor temperature "
        "falls, so sizing for the coldest day matters, and mild flow "
        "temperatures (larger radiators or underfloor loops) keep efficiency "
        "high. Because electricity is pricier per kWh than gas, running cost "
        "only beats a gas boiler when the system is well designed and the "
        "home is adequately insulated.",
    ),
    (
        "battery",
        "Home battery storage",
        "A home battery stores electricity when it is cheap or abundant and "
        "releases it when it is dear. Paired with rooftop solar it lifts "
        "self-consumption from around 30% to 70% or more by capturing midday "
        "surplus that would otherwise be exported for pennies. On a "
        "time-of-use tariff it charges off-peak overnight and discharges "
        "during the evening peak. Usable capacity of 5-13 kWh covers a typical "
        "evening. The dominant wear mechanism is cycle degradation: lithium "
        "iron phosphate cells tolerate daily cycling for a decade or more but "
        "slowly lose capacity.",
    ),
    (
        "evcharging",
        "Home EV charging",
        "Charging an electric car at home uses a wallbox of 3.7 to 22 kW; a "
        "7 kW unit adds roughly 25-30 miles of range per hour. A dedicated "
        "circuit is required, and an electrician must certify the "
        "installation. Smart chargers schedule sessions into the cheapest "
        "overnight window on a time-of-use tariff and can pause when the "
        "house draw is high, avoiding a supply upgrade. Solar owners can "
        "match charging to surplus generation on sunny afternoons. Cables "
        "tethered or socketed, app controls and RFID locking are convenience "
        "choices, not efficiency ones.",
    ),
    (
        "grid",
        "Grid balancing and demand response",
        "The electricity grid must match generation to consumption second by "
        "second. As wind and solar rise, balancing gets harder because their "
        "output is weather-driven, so the system operator pays for "
        "flexibility: batteries, idle plant and demand-side response. "
        "Domestic flexibility means shifting load - EV charging, water "
        "heating, heat-pump cycles - away from the evening peak, often "
        "automatically through tariffs or appliance APIs. Peak shaving on the "
        "distribution network defers expensive substation upgrades. "
        "Participants are paid per kWh shifted or per event accepted.",
    ),
    (
        "meter",
        "Smart meters",
        "A smart meter sends half-hourly consumption readings to the supplier "
        "over a dedicated radio network, ending estimated bills. The paired "
        "in-home display shows live spend in currency, which reliably cuts "
        "usage a few percent simply by making it visible. Half-hourly data is "
        "the enabler for time-of-use tariffs: without it a supplier cannot "
        "bill an off-peak window. Consumers can request the full export "
        "history and grant third parties read access. Opt-outs exist, and "
        "meters can be put back into credit-only mode without losing supply.",
    ),
    # -- engineering --------------------------------------------------------
    (
        "fatigue",
        "Fatigue crack growth",
        "Most mechanical failures start as a small crack that grows a little "
        "on every load cycle until the remaining section can no longer carry "
        "the stress. Growth per cycle, da/dN, follows the Paris law: it scales "
        "with the stress intensity factor range delta-K raised to a material "
        "exponent of roughly 3-4. Below a threshold range cracks are dormant; "
        "above it growth accelerates exponentially until the stress intensity "
        "reaches fracture toughness K_IC and the part fails fast. The stress "
        "ratio R (minimum over maximum load) shifts the curve, as do "
        "environment, weld residual stress and surface finish. Inspection "
        "intervals are set by integrating the curve from detectable flaw size "
        "to critical size with margin.",
    ),
    (
        "control",
        "PID loop tuning and windup",
        "A PID controller sums three actions on the error between setpoint "
        "and measurement: proportional gain reacts to present error, integral "
        "action removes steady-state offset, derivative action damps future "
        "error. Integral windup is the classic failure: when an actuator "
        "saturates at full opening, the integral term keeps accumulating, so "
        "the response overshoots badly when the process recovers. Cures are "
        "clamping the integrator or computing it from the actual actuator "
        "position (back-calculation). Aggressive tuning shortens rise time "
        "but erodes phase margin until the loop oscillates; Ziegler-Nichols "
        "and lambda tuning trade response speed against stability. Well-tuned "
        "loops also respect valve travel limits to avoid stem wear.",
    ),
    (
        "compressor",
        "Centrifugal compressor surge",
        "A centrifugal compressor raises pressure by accelerating gas through "
        "an impeller into a diffuser. On its performance map, the operating "
        "point sits between two stability boundaries: the surge line at low "
        "flow and choke at high flow. Crossing the surge line reverses flow "
        "momentarily, which can trip the machine or fatigue the blading "
        "within cycles. When downstream demand falls, antisurge control "
        "recycles gas from discharge to suction through a recycle valve so "
        "the point stays clear of the line. Speed, molecular weight and "
        "inlet temperature all move the map; Reynolds and Mach effects shift "
        "efficiency. Bearing loads during a surge event also drive mechanical "
        "limits on how close operation may approach the boundary.",
    ),
    (
        "fea",
        "Finite element mesh convergence",
        "A finite element model only approximates reality, and discretisation "
        "error must be bounded before trusting a stress result. An h-refinement "
        "study halves element size while watching a quantity of interest, "
        "such as peak von Mises stress: a converged result stops changing "
        "between successive meshes. Sharp re-entrant corners are stress "
        "singularities whose computed value grows forever as elements shrink, "
        "so geometry is filleted or the location excluded. Element aspect "
        "ratio, integration order (p-refinement) and boundary condition "
        "placement matter as much as size. Reporting a stress without a "
        "convergence check is the most common way analysis quietly goes "
        "wrong.",
    ),
    # -- finance ------------------------------------------------------------
    (
        "duration",
        "Bond duration and convexity",
        "Duration measures a bond's price sensitivity to yield: modified "
        "duration of 7 means a 1% yield rise drops the clean price by "
        "roughly 7%. Macaulay duration is the weighted average time to "
        "receive cash flows; zero-coupon bonds have duration equal to "
        "maturity, coupon bonds shorter. The linear estimate overshoots "
        "because the price-yield curve bends, and convexity corrects the "
        "second-order term, always in the holder's favour for plain bonds. "
        "Liability-driven investors immunise a portfolio by matching the "
        "duration of assets to obligations, then monitor drift as yields "
        "move and time passes. Yield to maturity at purchase anchors all of "
        "these sensitivities.",
    ),
    (
        "var",
        "Value at risk and expected shortfall",
        "One-day 99% value at risk of 2 million says the portfolio loses "
        "more than that on only 1% of days - it says nothing about how much "
        "more. Expected shortfall (conditional VaR) answers that: the average "
        "loss across the worst 1% of outcomes, which makes it coherent under "
        "diversification while VaR is not. Estimation choices matter: "
        "historical simulation resamples past returns, parametric methods "
        "assume a distribution, Monte Carlo simulates a model. Backtesting "
        "counts exceptions - days that breached the limit - against the "
        "binomial expectation; too many means the model understates tail "
        "risk, and regulators multiply capital penalties accordingly.",
    ),
    (
        "xva",
        "Collateral agreements and xVA",
        "A derivative's value depends on who bears default and funding risk. "
        "Under a collateral agreement the two sides post variation margin "
        "daily to mark-to-market, so counterparty exposure collapses to "
        "one day of moves; the discount rate becomes the overnight index "
        "swap curve because collateral cash earns the risk-free overnight "
        "rate. Valuation adjustments stack on top: credit (CVA) for expected "
        "counterparty default, debit (DVA) mirrored own default, and funding "
        "(FVA) for uncollateralised trades that consume capital. Initial "
        "margin covers the gap between trades and their cures; rehypothecation "
        "lets received collateral be reused. Ignoring these adjustments "
        "mis-prices every long-dated trade.",
    ),
    (
        "execution",
        "Execution benchmarks and slippage",
        "Institutional orders are sliced across time to avoid signalling. "
        "The standard measures of implementation cost compare achieved prices "
        "to a reference: slippage versus arrival price (implementation "
        "shortfall) captures both explicit spread paid and the price drift "
        "the order itself caused. Trading passively against a VWAP benchmark "
        "hides impact but risks missing the fill entirely on fast markets. "
        "A market order crosses the bid-ask spread immediately; a limit "
        "order earns the spread but suffers adverse selection when it fills. "
        "Depth in the limit order book decays quickly with size, so "
        "participation rates above a fifth of volume move price against the "
        "buyer. The art is buying liquidity only when momentum justifies it.",
    ),
    # -- medicine -----------------------------------------------------------
    (
        "sepsis",
        "Sepsis recognition and resuscitation",
        "Sepsis is life-threatening organ dysfunction from a dysregulated "
        "response to infection; a rising SOFA score or quick bedside criteria "
        "(qSOFA: respiratory rate, altered mentation, hypotension) flag it. "
        "Hyperlactataemia marks hypoperfusion, and lactate clearance over the "
        "first hours tracks resuscitation. Care bundles run within an hour: "
        "measure lactate, take blood cultures before broad-spectrum "
        "antibiotics, give 30 ml/kg crystalloid for hypotension or lactate "
        "above 4 mmol/L, start vasopressors if fluids fail, and hunt the "
        "source - an undrained abscess or obstructed biliary tree will not "
        "respond to antibiotics alone. Persistently high lactate after "
        "adequate fluids means escalation, not more fluid.",
    ),
    (
        "heartfailure",
        "Heart failure with reduced ejection fraction",
        "When the left ventricle ejects 40% or less of its volume (reduced "
        "ejection fraction), four drug "
        "classes each independently cut mortality and together form guideline "
        "medical therapy: a renin-angiotensin blocker (ARNI or ACE inhibitor), "
        "a beta-blocker, a mineralocorticoid receptor antagonist, and an "
        "SGLT2 inhibitor. Start all four early at low dose and titrate, "
        "rather than maximizing one at a time. NT-proBNP supports diagnosis "
        "and tracks response; NYHA class grades symptoms and guides device "
        "candidacy. Under-dosing, not drug choice, is the most common reason "
        "patients stay symptomatic. Potassium and renal function are checked "
        "as the MRA and SGLT2 doses advance.",
    ),
    (
        "anticoagulation",
        "Direct oral anticoagulants",
        "Direct oral anticoagulants - dabigatran (direct thrombin inhibitor) "
        "and the factor Xa inhibitors apixaban and rivaroxaban - need no "
        "routine INR monitoring, unlike warfarin, because their "
        "anticoagulant effect is predictable at fixed doses. Renal excretion "
        "varies: about 80% for dabigatran versus roughly a quarter for "
        "apixaban, so advanced chronic kidney disease pushes the choice "
        "toward factor Xa agents or warfarin. CHA2DS2-VASc scores the stroke "
        "risk that justifies treatment in atrial fibrillation; HAS-BLED "
        "flags bleeding risk to manage, not to withhold therapy. Specific "
        "reversal agents exist (idarucizumab for dabigatran, andexanet for "
        "Xa inhibitors). Gastrointestinal bleeding risk differs between "
        "agents and doses.",
    ),
    (
        "ckd",
        "Diabetic kidney disease",
        "Diabetes damages glomeruli slowly: albuminuria (urine "
        "albumin-to-creatinine ratio, UACR, above 30 mg/g) appears years before the "
        "estimated GFR falls below 60, and the combination stages chronic "
        "kidney disease. Two drug classes change the trajectory beyond "
        "glucose control: RAAS blockade (ACE inhibitor or ARB) lowers "
        "intraglomerular pressure and albuminuria, and SGLT2 inhibitors "
        "reduce hyperfiltration and cardiovascular death even in patients "
        "without diabetes. Hyperkalaemia limits up-titration - address it "
        "with dietary counselling or binders rather than abandoning "
        "renoprotection. Refer to nephrology when eGFR falls below 30, "
        "albuminuria is heavy, or decline exceeds 5 ml/min per year, since "
        "transplant work-up needs lead time.",
    ),
]

QUERIES = SEMANTIC_QUERIES = [
    # home energy
    ("how long until rooftop panels pay back their cost", "solar", False),
    ("storing cheap night electricity to use in the evening", "battery", False),
    ("cutting heat loss through the loft and walls", "insulation", False),
    ("is a backyard turbine noisy and does it need planning consent", "wind", False),
    ("how efficient is an air source unit on the coldest days", "heatpump", False),
    ("cheapest overnight window for charging an electric car", "evcharging", False),
    ("shifting household load when the network is stressed", "grid", False),
    ("seeing my half-hourly consumption and live spend", "meter", False),
    ("keeping midday surplus generation instead of exporting it", "battery", True),
    ("which heating technology moves heat instead of burning fuel", "heatpump", True),
    # engineering
    ("crack length growth per cycle above the threshold stress range", "fatigue", False),
    ("stop the integral action accumulating while the actuator is saturated", "control", False),
    ("recycle valve opening when flow demand approaches the instability line", "compressor", False),
    ("checking the stress result stops changing as elements shrink", "fea", False),
    (
        "bearing loads when the machine reverses flow past the stability boundary",
        "compressor",
        True,
    ),
    ("critical stress intensity where the part fails in one load cycle", "fatigue", False),
    # finance
    ("price drop for a one percent rise in yields", "duration", False),
    ("average loss across the worst one percent of days", "var", False),
    ("why discount at the overnight swap curve when margin is posted daily", "xva", False),
    ("cost of trading measured against the price when the order was released", "execution", False),
    (
        "adjustment charged for expected counterparty default on uncollateralised trades",
        "xva",
        False,
    ),
    ("spread captured by resting passively instead of crossing the book", "execution", True),
    # medicine
    ("lactate still high after fluid boluses what next", "sepsis", False),
    (
        "four drug classes that each reduce mortality in low ejection fraction",
        "heartfailure",
        False,
    ),
    ("anticoagulation without routine blood monitoring of the INR", "anticoagulation", False),
    ("urine protein ratio that predicts glomerular damage years ahead", "ckd", False),
    (
        "renal excretion makes which agent risky in severe kidney impairment",
        "anticoagulation",
        True,
    ),
    ("drug class that protects the kidney by lowering filtration pressure", "ckd", False),
]

# Short keyword queries; every token appears in the expected doc, so the
# library's AND-all-tokens FTS5 branch can and should match these.
LEXICAL_QUERIES = [
    ("U-values mineral wool", "insulation", False),
    ("coefficient of performance", "heatpump", False),
    ("half-hourly readings in-home display", "meter", False),
    ("Paris law stress intensity", "fatigue", False),
    ("integral windup back-calculation", "control", False),
    ("surge line recycle valve", "compressor", False),
    ("h-refinement convergence von Mises", "fea", False),
    ("modified duration convexity", "duration", False),
    ("expected shortfall backtesting exceptions", "var", False),
    ("variation margin rehypothecation", "xva", False),
    ("implementation shortfall arrival price", "execution", False),
    ("qSOFA lactate clearance", "sepsis", False),
    ("NT-proBNP ejection fraction", "heartfailure", False),
    ("idarucizumab reversal", "anticoagulation", False),
    ("albuminuria UACR hyperkalaemia", "ckd", False),
]
