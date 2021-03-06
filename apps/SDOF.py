'''
This is the SDOF file containing the layouts and functions related to the SDOF module.
'''


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px

from validator import *
from app import app

header = html.H3(
    'Single Degree Of Freedom',
    className=" mt-2, text-center")

about_Text = html.P(["This SDOF solver takes in your parameters and then produces a time history plot of your system. "
                    "Try it out by changing the input parameters and pressing submit to view your solution at the "
                     "bottom of the page. To submit feedback for this module please click ",
                     html.A("here",
                            href="https://forms.gle/puL3mKPbchXzsRrV7",
                            target="_blank"),
                     "."])

damp_switch = dbc.FormGroup(
    [
        dbc.Checklist(
            options=[
                {"label": "Use Damping Coefficient", "value": 1}
            ],
            value=[],
            id="damping-switch",
            switch=True,
        ),
    ]
)


''' The Following are the all the popover components. 
They consist of the blue ? button, The popover itself with the header and the validation message 
telling the user why/why not the current input is valid'''

mass_popover = html.Div(
    [
        dbc.Button("?",
                   id="mass-popover-target",
                   color="info",
        ),
        dbc.Popover(
            [
            dbc.PopoverHeader("Mass Input"),
            dbc.PopoverBody([], id="mass_validation_message"),
            ],
            id="mass_popover",
            is_open=False,
            target="mass-popover-target",
        ),
    ],
)

springConst_popover = html.Div(
    [
        dbc.Button(
            "?", id="springConst-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Sprint Constant Input"),
                dbc.PopoverBody([], id="springConst_validation_message"),
            ],
            id="springConst_popover",
            is_open=False,
            target="springConst-popover-target",
        ),
    ],
)

dampRatio_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampRatio-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Ratio Input"),
                dbc.PopoverBody([], id="dampRatio_validation_message"),
            ],
            id="dampRatio_popover",
            is_open=False,
            target="dampRatio-popover-target",
        ),
    ],
)

dampCoeff_popover = html.Div(
    [
        dbc.Button(
            "?", id="dampCoeff-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Damping Coefficient Input"),
                dbc.PopoverBody([], id="dampCoeff_validation_message"),
            ],
            id="dampCoeff_popover",
            is_open=False,
            target="dampCoeff-popover-target",
        ),
    ],
)

initialDisplacement_popover = html.Div(
    [
        dbc.Button(
            "?", id="initialDisplacement-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Initial Displacement Input"),
                dbc.PopoverBody([], id="initialDisplacement_validation_message"),
            ],
            id="initialDisplacement_popover",
            is_open=False,
            target="initialDisplacement-popover-target",
        ),
    ],
)


timeSpan_popover = html.Div(
    [
        dbc.Button(
            "?", id="timeSpan-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Time Span Input"),
                dbc.PopoverBody([], id="timeSpan_validation_message"),
            ],
            id="timeSpan_popover",
            is_open=False,
            target="timeSpan-popover-target",
        ),
    ],
)


numPts_popover = html.Div(
    [
        dbc.Button(
            "?", id="numPts-popover-target", color="info",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Number of Points Input"),
                dbc.PopoverBody([], id="numPts_validation_message"),
            ],
            id="numPts_popover",
            is_open=False,
            target="numPts-popover-target",
        ),
    ],
)


''' This is the set of inputs for the 1st system. It was done like this to accommodate multiple system inputs.'''
system1_input = dbc.Row([

    dbc.Col(
        html.Img(src=app.get_asset_url('SDOF_Pic.png'),
                 className="img-fluid"),
        className="col-12 col-sm-5 col-md-3 col-lg-3"),
    dbc.Col([
        dbc.Row(html.H6("System 1")),
        dbc.Row([
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon(
                        "Mass, m (kg)",
                        addon_type="prepend"
                    ),
                    dbc.Input(
                        id="m",
                        placeholder="kg",
                        debounce=True, type="number",
                        value=1, min=0.001, step=0.001),
                    dbc.InputGroupAddon(
                        mass_popover,
                        addon_type="append"
                    ),
                ],
            ), className="mb-1 col-12 col-md-12 col-lg-6"),

            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Spring Constant, k (N/m)", addon_type="prepend"),
                    dbc.Input(id="k", placeholder="N/m",
                              debounce=True, type="number",
                              value=1000, min=0.001, step=0.001),
                    dbc.InputGroupAddon(
                        springConst_popover,
                        addon_type="append"
                    ),
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),

            dbc.Col(damp_switch, width=12),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Ratio, ??", addon_type="prepend"),
                    dbc.Input(id="dampRatio", placeholder="",
                              debounce=True, type="number",
                              value=0.1, min=0, max=2, step=0.001),
                    dbc.InputGroupAddon(
                        dampRatio_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Damping Coefficient, c (Ns/m)", addon_type="prepend"),
                    dbc.Input(id="c", placeholder="Ns/m",
                              debounce=True, type="number",
                              value=1, min=0, step=0.001),
                    dbc.InputGroupAddon(
                        dampCoeff_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.H6("Initial Conditions"),
                className="mb-1 mt-1 col-12 col-sm-12 col-md-12 col-lg-12"
            ),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Initial Displacement, X0 (m)", addon_type="prepend"),
                    dbc.Input(id="x0", placeholder="m",
                              debounce=True, type="number",
                              value=0.1, min=-10, max=10, step=0.001),
                    dbc.InputGroupAddon(
                        initialDisplacement_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.H6("Computational Parameters"),
                className="mb-1 mt-1 col-12 col-sm-12 col-md-12 col-lg-12"
            ),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Time Span, t (s)", addon_type="prepend"),
                    dbc.Input(id="tend", placeholder="s",
                              debounce=True,  type="number",
                              value=2, min=0.01, max=360, step=0.01),
                    dbc.InputGroupAddon(
                        timeSpan_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(id="n", placeholder="",
                              debounce=True,  type="number",
                              min=10, step=1, value=1000),
                    dbc.InputGroupAddon(
                        numPts_popover,
                        addon_type="append"
                    )
                ],
            ), className="mb-1 col-12 col-sm-12 col-md-12 col-lg-6"),
            dbc.Col(
                html.P(id="aliasing_Warning", className="text-danger"),
                width=12
            ),
            dbc.Button("Submit", color="secondary", id='submit-button-state', size="sm")
        ]),

        dbc.Row(html.P(id="input_warning_string", className="text-danger")),
        dbc.Row(html.P(id="system_params")),

    ]),

], className="jumbotron")



'''This is where the layout is formed from the components.'''
layout = dbc.Container([
    header,
    about_Text,
    system1_input,
    html.H3("Time history plot of your solution", className=" mt-1 mb-1 text-center"),
    dcc.Graph(id='SDOF_plot', figure={}),

], fluid=True)




# ALL APP CALLBACKS
'''Here are all the app callbacks used. Note it is important that the id matches up to the Output/Input name'''


# ALL INPUT VALIDATORS and popover functions
# Each input has it's own validator passing in it's desired values to be checked against.
# They also have their own popover toggle function
@app.callback(
    Output("mass_validation_message", "children"),
    Output("mass-popover-target", "n_clicks"),
    Input("m", "value")
)
def mass_input_validator(mass_input):
    err_string, is_invalid = validate_input("mass", mass_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle mass popover with button (or validator callback above!!)
# Note the n_clicks is set as an Input to ensure updating if changed
@app.callback(
    Output("mass_popover", "is_open"),
    [Input("mass-popover-target", "n_clicks")],
    [State("mass_popover", "is_open")],
)
def mass_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("springConst_validation_message", "children"),
    Output("springConst-popover-target", "n_clicks"),
    Input("k", "value")
)
def springConst_input_validator(springConst_input):
    err_string, is_invalid = validate_input("spring constant", springConst_input, step=0.001, min=0.001)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle springConst popover with button (or validator callback above!!)
@app.callback(
    Output("springConst_popover", "is_open"),
    [Input("springConst-popover-target", "n_clicks")],
    [State("springConst_popover", "is_open")],
)
def springConst_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("dampRatio_validation_message", "children"),
    Output("dampRatio-popover-target", "n_clicks"),
    Input("dampRatio", "value")
)
def dampRatio_input_validator(dampRatio_input):
    err_string, is_invalid = validate_input("damping ratio", dampRatio_input, step=0.001, min=0, max=2)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampRatio popover with button (or validator callback above!!)
@app.callback(
    Output("dampRatio_popover", "is_open"),
    [Input("dampRatio-popover-target", "n_clicks")],
    [State("dampRatio_popover", "is_open")],
)
def dampRatio_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("dampCoeff_validation_message", "children"),
    Output("dampCoeff-popover-target", "n_clicks"),
    Input("c", "value")
)
def dampCoeff_input_validator(dampCoeff_input):
    err_string, is_invalid = validate_input("damping coefficient", dampCoeff_input, step=0.001, min=0)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle dampCoeff popover with button (or validator callback above!!)
@app.callback(
    Output("dampCoeff_popover", "is_open"),
    [Input("dampCoeff-popover-target", "n_clicks")],
    [State("dampCoeff_popover", "is_open")],
)
def dampCoeff_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("initialDisplacement_validation_message", "children"),
    Output("initialDisplacement-popover-target", "n_clicks"),
    Input("x0", "value")
)
def initialDisplacement_input_validator(initialDisplacement_input):
    err_string, is_invalid = validate_input("initial displacement", initialDisplacement_input, step=0.001, min=-10, max=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle initialDisplacement popover with button (or validator callback above!!)
@app.callback(
    Output("initialDisplacement_popover", "is_open"),
    [Input("initialDisplacement-popover-target", "n_clicks")],
    [State("initialDisplacement_popover", "is_open")],
)
def initialDisplacement_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("timeSpan_validation_message", "children"),
    Output("timeSpan-popover-target", "n_clicks"),
    Input("tend", "value")
)
def timeSpan_input_validator(timeSpan_input):
    err_string, is_invalid = validate_input("time span", timeSpan_input, step=0.01, min=0.01, max=360)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle timeSpan popover with button (or validator callback above!!)
@app.callback(
    Output("timeSpan_popover", "is_open"),
    [Input("timeSpan-popover-target", "n_clicks")],
    [State("timeSpan_popover", "is_open")],
)
def timeSpan_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    Output("numPts_validation_message", "children"),
    Output("numPts-popover-target", "n_clicks"),
    Input("n", "value")
)
def numPts_input_validator(numPts_input):
    err_string, is_invalid = validate_input("number of points", numPts_input, step=1, min=10)
    if is_invalid:
        return err_string, 1    # Set nclicks to 1 to call popover toggle
    else:
        return err_string, 0    # Set nclicks to 0 to prevent popover


# Toggle numPts popover with button (or validator callback above!!)
@app.callback(
    Output("numPts_popover", "is_open"),
    [Input("numPts-popover-target", "n_clicks")],
    [State("numPts_popover", "is_open")],
)
def numPts_toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


# ======= ALIASING WARNING ==================
@app.callback(
    Output("aliasing_Warning", "children"),
    [Input("m", "value"),
     Input("k", "value"),
     Input("tend", "value"),
     Input("n", "value"),
     ]
)
def aliasing_check(m, k, timeSpan, nPts):
    aliasing_warning = validate_aliasing(m, k, timeSpan, nPts)
    return aliasing_warning





# ======== Damping Ratio & Coefficient Updates =============

# This function disables the damping ratio or damping coefficient input using the toggle
@app.callback(
    Output("dampRatio", "disabled"),
    Output("c", "disabled"),
    Input("damping-switch", "value")
)
def damping_toggle(switch):
    switch_state = len(switch)
    return switch_state, not switch_state



# This function updates damping coefficient c when it is disabled and other values are inputted
@app.callback(
    Output(component_id='c', component_property='value'),
    Input(component_id='c', component_property='disabled'),
    Input(component_id='c', component_property='value'),
    Input(component_id='dampRatio', component_property='value'),
    Input(component_id='k', component_property='value'),
    Input(component_id='m', component_property='value')
)
def update_c(c_disabled, c, dampRatio, k, m):
    if c_disabled and m!=None and k!=None and dampRatio!=None:
        c = np.round((dampRatio * 2 * np.sqrt(k * m)),3)
    return c


# This function updates damping ratio when it is disabled and other values are inputted
@app.callback(
    Output(component_id='dampRatio', component_property='value'),
    Input(component_id='dampRatio', component_property='disabled'),
    Input(component_id='dampRatio', component_property='value'),
    Input(component_id='c', component_property='value'),
    State(component_id='k', component_property='value'),
    State(component_id='m', component_property='value')
)
def update_damping_ratio(dampRatio_disabled, dampRatio, c, k, m):
    if dampRatio_disabled and m!=None and k!=None and c!=None:
        dampRatio = np.round((c / (2 * np.sqrt(k * m))),3)
    return dampRatio






# ============ Plotting Graph ========================

'''
This update output function takes in the inputs, passes it to the validator and solver and then updates the graphs/text
using the solutions/values it gets back.
  Inputs
  n_clicks, m, k, dampRatio, dampCoeff, x0, tend, n
      n_clicks (num) = number of clicks of the submit button (NOTE: This is used as the only Input in order to update 
                       the outputs when the submit button is clicked.)  
      m (num) = mass (kg)
      k (num) = Stiffness constant (N/m)
      dampRatio (num) = The Damping Ratio
      dampCoeff (num) = The Damping Coefficient (Ns/m) 
      x0 (num) = The initial displacement of the mass (m)
      tend (num) = The time to perform the calculation upto (s)
      n (num) = The number of points to use for the plot/array
  Output
     fig, input_warning_string, system_params
     fig (Figure object) = The object that holds all the parameters to produce the figure
     input_warning_string (string) = A string that returns any warning messages if any in red 
     system_params (string array) = An array which concatenates any important parameters of the system to be 
                                    displayed as text
'''
@app.callback(Output('SDOF_plot', 'figure'),
              Output('input_warning_string', 'children'),
              Output('system_params', 'children'),
              Input('submit-button-state', 'n_clicks'),
              State('m', 'value'),
              State('k', 'value'),
              State('dampRatio', 'value'),
              State('c', 'value'),
              State('x0', 'value'),
              State('tend', 'value'),
              State('n', 'value'))
def update_output(n_clicks, m, k, dampRatio, dampCoeff, x0, tend, n):

    # First validate inputs
    is_invalid = validate_all_inputsSDOF(m,k,dampRatio, dampCoeff, x0, tend, n)

    if(is_invalid):
        # Reset Graphs
        fig = px.line(x=[0], y=[0],
                      labels=dict(
                          x="Time (sec)",
                          y="Displacement, x (m)"
                      )
                      )
        input_warning_string = ["Graph was cleared!",
                                html.Br(),
                                "Please check your inputs before Submitting again!"]
        system_params = [""]
        return fig, input_warning_string, system_params
    else:
        x, t, wn, wnHz, wd, wdHz, maxAmp, solutionType = SDOF_solver(m, k, dampRatio, x0, tend, n)
        fig = px.line(x=t, y=x,
                      labels=dict(
                          x="Time (sec)",
                          y="Displacement, x (m)"
                      )
                      )
        input_warning_string = ""
        # Create a string here!!!!!! Make solver function spit out the frequency, Hz and rad/s and max amplitude!!! ====================================

        if 0 < dampRatio < 1:
            # If the system is underdamped there will be damped natural freq
            dampedNatFreq_string = ["Damped Natural Frequency, ??d (rad/s): " + str(wd) + " rad/s", html.Br(),
                                    "Damped Natural Frequency, ??d (Hz): " + str(wdHz) + " Hz", html.Br(), ]
        else:
            # Otherwise no damped nat frequ
            dampedNatFreq_string = [""]

        system_params = ["Please scroll down to see your solution.", html.Br(), html.Br(),
                         "System Parameters:", html.Br(),
                         "Natural Frequency, ??n (rad/s): "+ str(wn) + " rad/s", html.Br(),
                         "Natural Frequency, ??n (Hz): " + str(wnHz) + " Hz", html.Br(),
                         "Maximum displacement (m): " + str(maxAmp) + " m", html.Br(),
                         solutionType, html.Br()
                         ] + dampedNatFreq_string

        return fig, input_warning_string, system_params

'''
This is the actual SDOF Solver which takes in the inputs and returns a solution along with any significant values
  Inputs
      m (num) = Mass (kg)
      k (num) = Stiffness constant (N/m)
      dampRatio (num) = The Damping Ratio 
      x0 (num) = The initial displacement of the mass (m)
      tend (num) = The time to perform the calculation upto (s)
      n (num) = The number of points to use for the plot/array
  Output
      x (array) = The displacement solution over time (m)
      t (array) = The time array for the solution (s)
      wn (num) = The natural frequency of the system (rad/s)
      wnHz (num) = The natural frequency of the system in Hz (Hz)
      wd (num) = The damped natural frequency of the system (rad/s)
      wd (num) = The damped natural frequency of the system in Hz (Hz)
      maxAmp (num) = The maximum amplitude of the system response (m)
      solutionType (string) = A string telling the user what type of solution the response is
'''
def SDOF_solver(m, k, dampRatio, x0, tend, n):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    wnHz = wn/(2*np.pi)     # Natural freq in Hz
    if 0 < dampRatio < 1:
        wd = wn*np.sqrt(1-dampRatio**2)     # Damped nat freq (rad/s)
        wdHz = wd/(2*np.pi)     # Damped Natural freq in Hz
    else:
        wd = 0
        wdHz = 0

    t = np.linspace(0, tend, n)


    if dampRatio == 0:
        x = x0 * np.cos(wn * t)
        solutionType = "This is an Undamped Solution"
    elif 1 > dampRatio > 0:
        solutionType = "This is an Under Damped Solution"
        wd = wn * np.sqrt(1 - dampRatio ** 2)
        A = x0
        B = dampRatio * A / wd
        x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
    elif dampRatio == 1:
        solutionType = "This is a Critically Damped Solution"
        A = x0
        B = A * wn
        x = (A + B * t) * np.exp(-wn * t)
    elif dampRatio > 1:
        solutionType = "This is an Over Damped Solution"
        A = x0 * (dampRatio + np.sqrt(dampRatio ** 2 - 1)) / (2 * np.sqrt(dampRatio ** 2 - 1))
        B = x0 - A
        x = A * np.exp((-dampRatio + np.sqrt(dampRatio ** 2 - 1)) * wn * t) + B * np.exp(
            (-dampRatio - np.sqrt(dampRatio ** 2 - 1)) * wn * t)
    else:
        solutionType = "This is an unaccounted for Solution"

    maxAmp = np.round(max(x), decimals=2)

    return x, t, np.round(wn,decimals=2), np.round(wnHz,decimals=2), np.round(wd,decimals=2), np.round(wdHz,decimals=2), maxAmp, solutionType