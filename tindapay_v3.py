import base64
import io
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, ALL
import plotly.express as px
import plotly.graph_objects as go 
import pandas as pd

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


app.layout = html.Div([
    html.H1('Upload TindaPay Dashboard Data', style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload', children=[]),
    html.Div(id='outlet-data-container', children=[
        html.H2('Outlet Stores Performance', style={'textAlign': 'center', 'width': '100%'}),
        html.Div([
            html.Label('Select PJP:'),
            dcc.Dropdown(
                id='pjp-filter',
                options=[],
                value=[],  # Default value is all PJP
                multi=True
            ),
            html.Label('Select Channel:'),
            dcc.Dropdown(
                id='channel-filter',
                options=[],
                value=[],  # Default value is all channels
                multi=True
            ),
        ]),
        # dash_table.DataTable(
        #     id='outlet-table',
        #     columns=[
        #         {'name': 'Outlet Code', 'id': 'Outlet Code'},
        #         {'name': 'Outlet Name', 'id': 'Outlet Name'},
        #         {'name': 'Amount Pending', 'id': 'Amount Pending'},
        #         {'name': 'Ageing', 'id': 'Ageing'}
        #     ],
        #     style_data_conditional=[
        #         {
        #             'if': {'filter_query': '{Ageing} < 7', 'column_id': 'Ageing'},
        #             'backgroundColor': 'green',
        #             'color': 'white'
        #         },
        #         {
        #             'if': {'filter_query': '{Ageing} = 7', 'column_id': 'Ageing'},
        #             'backgroundColor': 'yellow',
        #             'color': 'black'
        #         },
        #         {
        #             'if': {'filter_query': '{Ageing} >= 8', 'column_id': 'Ageing'},
        #             'backgroundColor': 'red',
        #             'color': 'white'
        #         },
        #     ],
        # ),
        html.Div(id='graphs-container')
    ], style={'display': 'none'}),
])

@app.callback(
    Output('output-data-upload', 'children'),
    Output('graphs-container', 'children'),
    Output('pjp-filter', 'options'),
    Output('channel-filter', 'options'),
    Output('outlet-data-container', 'style'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')],
    prevent_initial_call=True
)
def update_output(contents, filename, date):
    children = []
    pjp_options = set()
    channel_options = set()
    if contents is not None:
        for i, (c, n, d) in enumerate(zip(contents, filename, date)):
            content_type, content_string = c.split(',')
            try:
                decoded = base64.b64decode(content_string)
                if 'csv' in n:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in n:
                    df_usage = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='A:H')
                    df_repeat_v2 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='N:S')
                    df_repeat_v3 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='U:Y')
                    df_repayment_v1 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='AA:AF')
                    df_repayment_v2 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='AH:AM')
                    df_outlet =  pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='AO:AT')
                    df_gsv =  pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='AV:AZ')
                    df_buy1 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='BB:BF')
                    df_buy2 = pd.read_excel(io.BytesIO(decoded), sheet_name="TINDAPAY", skiprows=1, usecols='BH:BL')


                    # Rename columns to avoid conflicts
                    df_usage.columns = [col.strip() for col in df_usage.columns]
                    df_repeat_v2.columns = [col.strip() for col in df_repeat_v2.columns]
                    df_repeat_v3.columns = [col.strip() for col in df_repeat_v3.columns]
                    df_repayment_v1.columns= [col.strip() for col in df_repayment_v1.columns]
                    df_repayment_v2.columns= [col.strip() for col in df_repayment_v2.columns]
                    df_outlet.columns = [col.strip() for col in df_outlet.columns] 
                    df_gsv.columns = [col.strip() for col in df_gsv.columns] 
                    df_buy1.columns = [col.strip() for col in df_buy1.columns] 
                    df_buy2.columns = [col.strip() for col in df_buy2.columns] 


                    # Normalize column names to avoid conflicts
                    df_usage.columns = [col.split('.1')[0] for col in df_usage.columns]
                    df_repeat_v2.columns = [col.split('.1')[0] for col in df_repeat_v2.columns]
                    df_repeat_v3.columns = [col.strip('.2') for col in df_repeat_v3.columns]
                    df_repayment_v1.columns= [col.strip('.3') for col in df_repayment_v1.columns]
                    df_repayment_v2.columns= [col.strip('.4') for col in df_repayment_v1.columns]
                    df_outlet.columns = [col.strip('.5') for col in df_outlet.columns]
                    df_gsv.columns = [col.strip('.6') for col in df_gsv.columns]
                    df_buy1.columns = [col.strip('.7') for col in df_buy1.columns] 
                    df_buy1.columns = [col.strip('.1') for col in df_buy1.columns]     
                    df_buy2.columns = [col.strip('.8') for col in df_buy2.columns] 
                    df_buy2.columns = [col.strip('.2') for col in df_buy2.columns] 

                    print(df_repayment_v1)

                    # Ensure required columns are present
                    required_columns_usage = ['CHANNEL', 'PJP']
                    required_columns_repeat = ['CHANNEL', 'PJP']  # Adjust if needed

                    if not all(col in df_usage.columns for col in required_columns_usage):
                        return html.Div(['Missing required columns in USAGE sheet.']), [], [], [], {'display': 'none'}
                    if not all(col in df_repeat_v2.columns for col in required_columns_repeat):
                        return html.Div(['Missing required columns in REPEAT sheet.']), [], [], [], {'display': 'none'}
                    if not all(col in df_repeat_v3.columns for col in required_columns_repeat):
                        return html.Div(['Missing required columns in REPEAT sheet.']), [], [], [], {'display': 'none'}
                    if not all(col in df_repayment_v1.columns for col in required_columns_repeat):
                        return html.Div(['Missing required columns in REPAYMENT sheet.']), [], [], [], {'display': 'none'}

                    df_dict = {
                        '1. USAGE': df_usage,
                        '2. REPEAT': df_repeat_v2,
                        '3. REPEAT': df_repeat_v3,
                        '4. REPAYMENT': df_repayment_v1,
                        '5. REPAYMENT': df_repayment_v2,
                        '6. OUTLET': df_outlet,
                        '7. GSV': df_gsv,
                        '8. BUY': df_buy1,
                        '9. BUY': df_buy2
                    }
                else:
                    return html.Div(['Unsupported file format.']), [], [], [], {'display': 'none'}

                all_dfs = []
                for sheet_name, df in df_dict.items():
                    # Drop rows with null values in CHANNEL and PJP columns
                    df = df.dropna(subset=['CHANNEL', 'PJP'])

                    # Generate unique PJP and Channel options
                    pjp_options.update(df['PJP'].unique())
                    channel_options.update(df['CHANNEL'].unique())

                    # Store data for graphs
                    all_dfs.append(html.Div([
                        dcc.Store(id={'type': 'store-data', 'index': f'{i}-{sheet_name}'}, data=df.to_dict('records')),
                        dcc.Graph(
                            id={'type': 'dynamic-graph', 'index': f'{i}-{sheet_name}'},
                            figure=create_graph(df, sheet_name, df['PJP'].unique(), df['CHANNEL'].unique())
                        ),
                        html.Hr()
                    ]))

                return children, all_dfs, [{'label': pjp, 'value': pjp} for pjp in pjp_options], [{'label': channel, 'value': channel} for channel in channel_options], {'display': 'block'}
            except Exception as e:
                print(f"Error processing file {n}: {e}")
                return html.Div(['There was an error processing this file.']), [], [], [], {'display': 'none'}

    return "", [], [], [], {'display': 'none'}

@app.callback(
    Output({'type': 'dynamic-graph', 'index': ALL}, 'figure'),
    [Input('pjp-filter', 'value'),
     Input('channel-filter', 'value')],
    [State({'type': 'store-data', 'index': ALL}, 'data'),
     State({'type': 'dynamic-graph', 'index': ALL}, 'id')]
)
def update_graph(pjp_values, channel_values, stored_data, graph_ids):
    figures = []
    for i in range(len(graph_ids)):
        pjp = pjp_values
        channel = channel_values
        
        df = pd.DataFrame(stored_data[i])

        # Filter df based on selected PJP and Channel
        filtered_df = df[(df['PJP'].isin(pjp)) & (df['CHANNEL'].isin(channel))]

        # Create the updated graph
        fig = create_graph(filtered_df, graph_ids[i]['index'].split('-')[1], pjp, channel)
        figures.append(fig)

    return figures

def create_graph(df, sheet_name, selected_pjp, selected_channel):
    fig = None

    # Debug: Print the dataframe for inspection
    # print(df)
    
    # Ensure required columns are present
    if 'WK' in df.columns and 'USAGE VS. TOTAL PJP' in df.columns:
        # Filter data based on selected PJP and Channel
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]

        # Ensure 'USAGE VS. TOTAL PJP' and 'USAGE VS. TOTAL SAMPLE' are numeric
        filtered_df['USAGE VS. TOTAL PJP'] = pd.to_numeric(filtered_df['USAGE VS. TOTAL PJP'], errors='coerce')
        filtered_df['USAGE VS. TOTAL SAMPLE'] = pd.to_numeric(filtered_df['USAGE VS. TOTAL SAMPLE'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = go.Figure()

            # Adding the trace for USAGE VS. TOTAL PJP
            fig.add_trace(go.Scatter(
                x=filtered_df['WK'],
                y=filtered_df['USAGE VS. TOTAL PJP'],
                mode='lines+markers+text',
                name='Usage vs. Total PJP',
                text=filtered_df['USAGE VS. TOTAL PJP'].apply(lambda x: f"{x:.2f}"),
                textposition='top right',
                line=dict(color='blue'),
                marker=dict(size=8)
            ))

            # Adding the trace for USAGE VS. TOTAL SAMPLE
            fig.add_trace(go.Scatter(
                x=filtered_df['WK'],
                y=filtered_df['USAGE VS. TOTAL SAMPLE'],
                mode='lines+markers+text',
                name='Usage vs. Total Sample',
                text=filtered_df['USAGE VS. TOTAL SAMPLE'].apply(lambda x: f"{x:.2f}"),
                textposition='top right',
                line=dict(color='red'),
                marker=dict(size=8)
            ))

            # Updating the layout for titles and axis labels
            fig.update_layout(
                title="Usage Rate Over Weeks",
                xaxis_title="Week",
                yaxis_title="Value",
                legend_title="Metrics",
                hovermode="x unified"
            )
    

    elif 'REPEAT' in df.columns:
        # Handle REPEAT and NEW columns
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]
        
        # Ensure 'REPEAT' and 'NEW' are numeric
        filtered_df['REPEAT'] = pd.to_numeric(filtered_df['REPEAT'], errors='coerce')
        filtered_df['NEW'] = pd.to_numeric(filtered_df['NEW'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='WK.2',
                y=['REPEAT', 'NEW'],
                title="Weekly Repeat and New Counts",
                labels={'WK.2': 'Week', 'value': 'Count', 'variable': 'Type'},
                barmode='stack'
            )
            # Create text labels for each segment
            df_melted = df.melt(id_vars='WK.2', value_vars=['REPEAT', 'NEW'])
            text_labels = df_melted.groupby(['WK.2', 'variable'])['value'].apply(lambda x: [f"{v:.2f}" for v in x]).reset_index(name='text')
            fig.update_traces(text=df['REPEAT'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='REPEAT'))
            fig.update_traces(text=df['NEW'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='NEW'))
            # fig.update_traces(
            #     text=filtered_df[['REPEAT', 'NEW']].apply(lambda row: f"{row['REPEAT']:.2f}" if pd.notnull(row['REPEAT']) else '', axis=1),
            #     textposition='inside'
            # )

    elif 'ECOV' in df.columns:
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]

        # Create a figure with a table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(filtered_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[filtered_df[col] for col in filtered_df.columns],
                    fill_color='lavender',
                    align='left'))
        ])
        # Add a title to the table
        fig.update_layout(
            title={
                'text': "Repeat Rate View",  # Replace with your desired title
                'y': 0.95,  # Adjust the title's vertical position
                'x': 0.5,   # Center the title horizontally
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

    elif 'Paid' in df.columns:
        # Handle REPEAT and NEW columns
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]
        
        # Ensure 'REPEAT' and 'NEW' are numeric
        filtered_df['Paid'] = pd.to_numeric(filtered_df['Paid'], errors='coerce')
        filtered_df['Outstanding'] = pd.to_numeric(filtered_df['Outstanding'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='WK',
                y=['Paid', 'Outstanding'],
                title="Paid and Outstanding Balance",
                labels={'WK': 'Week', 'value': 'Amount', 'variable': 'Type'},
                barmode='stack'
            )
            # # Create text labels for each segment
            # df_melted = df.melt(id_vars='Paid', value_vars=['Paid', 'Outstanding'])
            # text_labels = df_melted.groupby(['Paid', 'Outstanding'])['value'].apply(lambda x: [f"{v:.2f}" for v in x]).reset_index(name='text')
            fig.update_traces(text=df['Paid'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Paid'))
            fig.update_traces(text=df['Outstanding'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Outstanding'))
            # fig.update_traces(
            #     text=filtered_df[['Paid', 'Outstanding']].apply(lambda row: f"{row['Paid']:.2f}" if pd.notnull(row['Paid']) else '', axis=1),
            #     textposition='inside'
            # )
    elif 'Outlet Name' in df.columns:
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]

        # Define colors for the Ageing cells based on the value
        ageing_colors = ['green' if val < 7 else 'yellow' if val == 7 else 'red' for val in filtered_df['Ageing']]

        # Create a figure with a table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(filtered_df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[filtered_df[col] for col in filtered_df.columns],
                fill_color=[
                    'lavender',  # For the first column
                    'lavender',  # For the second column
                    'lavender',  # For the third column
                    'lavender',  # For the fourth column
                    'lavender',  # For 'Amount Pending'
                    ageing_colors  # Color-coding for the 'Ageing' column
                ],
                align='left'
            )
        )])
         # Add a title to the table
        fig.update_layout(
            title={
                'text': "Outlet Stores Performance",  # Replace with your desired title
                'y': 0.95,  # Adjust the title's vertical position
                'x': 0.5,   # Center the title horizontally
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

    elif 'GSV (PHP)' in df.columns:
        # Handle REPEAT and NEW columns
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]
        
        # Ensure 'REPEAT' and 'NEW' are numeric
        filtered_df['GSV (PHP)'] = pd.to_numeric(filtered_df['GSV (PHP)'], errors='coerce')
        filtered_df['Growth vs Baseline'] = pd.to_numeric(filtered_df['Growth vs Baseline'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='Month',
                y=['GSV (PHP)', 'Growth vs Baseline'],
                title="GSV and Growth vs Baseline",
                labels={'Month': 'Month', 'value': 'GSV (PHP)', 'variable': 'Type'},
                barmode='stack'
            )
            fig.update_traces(text=df['GSV (PHP)'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='GSV (PHP)'))
            fig.update_traces(text=df['Growth vs Baseline'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Growth vs Baseline'))
            # fig.update_traces(
            #     text=filtered_df[['GSV (PHP)', 'Growth vs Baseline']].apply(lambda row: f"{row['GSV (PHP)']:.2f}" if pd.notnull(row['GSV (PHP)']) else '', axis=1),
            #     textposition='inside'
            # )
    elif 'No. of Invoices' in df.columns:
        # Handle REPEAT and NEW columns
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]
        
        # Ensure 'REPEAT' and 'NEW' are numeric
        filtered_df['No. of Invoices'] = pd.to_numeric(filtered_df['No. of Invoices'], errors='coerce')
        filtered_df['Growth vs Baseline'] = pd.to_numeric(filtered_df['Growth vs Baseline'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='Month',
                y=['No. of Invoices', 'Growth vs Baseline'],
                title="No. of Invoices and Growth vs Baseline",
                labels={'Month': 'Month', 'value': 'No. of Invoices', 'variable': 'Type'},
                barmode='stack'
            )
            fig.update_traces(text=df['No. of Invoices'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='No. of Invoices'))
            fig.update_traces(text=df['Growth vs Baseline'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Growth vs Baseline'))
            # fig.update_traces(
            #     text=filtered_df[['No. of Invoices', 'Growth vs Baseline']].apply(lambda row: f"{row['No. of Invoices']:.2f}" if pd.notnull(row['No. of Invoices']) else '', axis=1),
            #     textposition='inside'
            # )
    elif 'Grew vs. Baseline (No. of Doors)' in df.columns:
        # Handle REPEAT and NEW columns
        filtered_df = df[(df['PJP'].isin(selected_pjp)) & (df['CHANNEL'].isin(selected_channel))]
        
        # Ensure 'REPEAT' and 'NEW' are numeric
        filtered_df['Grew vs. Baseline (No. of Doors)'] = pd.to_numeric(filtered_df['Grew vs. Baseline (No. of Doors)'], errors='coerce')
        filtered_df['Did not grow vs. Baseline'] = pd.to_numeric(filtered_df['Did not grow vs. Baseline'], errors='coerce')

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='Month',
                y=['Grew vs. Baseline (No. of Doors)', 'Did not grow vs. Baseline'],
                title="Grew vs. Baseline (No. of Doors) and Did not grow vs. Baseline",
                labels={'Month': 'Month', 'value': 'No. of Outlets', 'variable': 'Type'},
                barmode='stack'
            )
            fig.update_traces(text=df['Grew vs. Baseline (No. of Doors)'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Grew vs. Baseline (No. of Doors)'))
            fig.update_traces(text=df['Did not grow vs. Baseline'].apply(lambda x: f"{x:.2f}"), textposition='inside', selector=dict(name='Did not grow vs. Baseline'))
            # fig.update_traces(
            #     text=filtered_df[['Grew vs. Baseline (No. of Doors)', 'Did not grow vs. Baseline']].apply(lambda row: f"{row['Grew vs. Baseline (No. of Doors)']:.2f}" if pd.notnull(row['Grew vs. Baseline (No. of Doors)']) else '', axis=1),
            #     textposition='inside'
            # )

    if fig is not None:
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Ensures every tick on x-axis is labeled
    else:
        # Create an empty figure if no valid data was found
        fig = go.Figure()
        fig.update_layout(title=f"{sheet_name}: Choose which Channel and PJP above")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
