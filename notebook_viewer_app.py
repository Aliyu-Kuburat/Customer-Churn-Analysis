
import streamlit as st
import nbformat
import base64
import io
import matplotlib.pyplot as plt

# Define the path to the notebook file
notebook_file = 'customer_churn_prediction.ipynb'

st.title('Customer Churn Prediction Notebook Viewer')

try:
    # Read the notebook file
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Iterate through each cell and render its content
    for cell in notebook_content.cells:
        if cell.cell_type == 'markdown':
            st.markdown(cell.source)
        elif cell.cell_type == 'code':
            st.code(cell.source, language='python')
            for output in cell.outputs:
                if output.output_type == 'stream':
                    st.text(output.text)
                elif output.output_type == 'display_data' or output.output_type == 'execute_result':
                    if 'image/png' in output.data:
                        image_data = output.data['image/png']
                        st.image(base64.b64decode(image_data), use_column_width=True)
                    elif 'image/jpeg' in output.data:
                        image_data = output.data['image/jpeg']
                        st.image(base64.b64decode(image_data), use_column_width=True)
                    elif 'text/plain' in output.data:
                        st.text(output.data['text/plain'])
                    # Add more output types as needed
except FileNotFoundError:
    st.error(f"Error: The notebook file '{notebook_file}' was not found. Please ensure it is saved in the same directory as this Streamlit app.")
except Exception as e:
    st.error(f"An error occurred: {e}")
