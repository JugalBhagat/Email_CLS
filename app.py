from flask import Flask ,render_template, request, send_file
import pandas as pd
import subprocess

app = Flask(__name__)
dataframe=None

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/api_use')
def api_useage():
    return render_template("api_inst.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        selected_model = request.form.get('model')
        uploaded_file = request.files['csvfile']
        print("info got")
        vali, dataframe = validate(uploaded_file)

        if vali != "":
            print(vali)
            return "Model: {}, CSVFile: {}, \n {},\n {}".format(selected_model, uploaded_file.filename, vali)
        else:
            ds_ham, ds_spam = predict(selected_model, uploaded_file, dataframe)
            return render_template("result.html", model=selected_model, csvfile=uploaded_file, ds_ham=ds_ham, ds_spam=ds_spam)
    except Exception as e:
        error_message = f"(UPload()) An error occurred during file upload and processing: {str(e)}"
        print(error_message)
        return render_template("error.html",error=str(e)) 
    
def validate(uploaded_file):
    vali = ""
    dataframe = None 
    try:
        if uploaded_file.filename == '':
            vali = "No selected file"
        else:
            df = pd.read_csv(uploaded_file)
            dataframe = df
            required_columns = ['index', 'text']
            if set(required_columns) != set(df.columns):
                vali = "The dataset does not have the required columns."
    except Exception as e:
        vali = f" (Validate) An error occurred during validation: {str(e)}"
        print(vali)
        return render_template("error.html",error=str(e)) 
    
    return vali, dataframe

def predict(selected_model, uploaded_file, dataframe):
    try:
        print(f"selected_model: {selected_model}")
        print(f"uploaded_file: {uploaded_file}")
        print(f"dataframe: {dataframe}")
        print("----end of predict-----")
        
        # Assuming that exceptions may be raised in the following functions
        from spaham_uploaded_classify import temp, predict2
        ds_ham, ds_spam = predict2(dataframe, selected_model)
        return ds_ham, ds_spam
    except Exception as e:

        print(str(e))
        return render_template("error.html",error=str(e))  
     
@app.route('/page2')
def page2():
    import os;
    file_value = request.args.get('file')
    # Determine the file path based on the 'file' parameter
    if file_value == 'ham_email':
        file_path = 'ham_email.csv'  # Replace with the actual file path for ham_email.csv
    elif file_value == 'spam_email':
        file_path = 'spam_email.csv'  # Replace with the actual file path for spam_email.csv
    else:
        return "Invalid file parameter"

    # Check if the file exists
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found"
    
@app.route('/prepro')
def data_prepro():
    return render_template("prepro_form.html")

@app.route('/preprocess', methods=['POST'])
def process_form():
    try:
        uploaded_file = request.files['csvfile']
        additional_columns = request.form.getlist('additionalColumn')
        import pandas as pd
        import os
        
        new_ds=pd.read_csv(uploaded_file,index_col=False)
        new_ds.fillna('', inplace=True)
        new_ds = new_ds.astype(str)
        new_ds['text'] = new_ds[additional_columns].astype(str).agg(' '.join, axis=1)
        new_ds = new_ds[['text']]
        new_ds.reset_index(drop=True, inplace=True)
        new_ds.columns = ['text']
        new_ds.index.name = 'index'



        #new_ds.reset_index(inplace=True, drop=False)
        #new_ds = new_ds.rename(columns={'^unnamed':'index'})
        #new_ds = new_ds.loc[:, ~new_ds.columns.str.contains('^Unnamed')]
        # new_ds = new_ds.rename(columns={'': 'index'})


        if os.path.exists("new_updated_dataset.csv"):
            os.remove("new_updated_dataset.csv")
        new_ds.to_csv("new_updated_dataset.csv")

        
        print(new_ds)
        if uploaded_file:
            uploaded_file.save(uploaded_file.filename)
        os.remove(uploaded_file.filename)
        
        #print(additional_columns[0])
    except Exception as e:
        return e
    
    return "Your new updated CSV file is saved as new_updated_dataset.csv"

if __name__ == '__main__':
    app.run(debug=True,port=8000)
