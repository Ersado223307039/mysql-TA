from flask import Flask, render_template, request ,jsonify ,redirect, url_for, session
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import joblib
import os
import requests
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)


# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'  
app.config['MYSQL_USER'] = 'root'       
app.config['MYSQL_PASSWORD'] = ''       
app.config['MYSQL_DB'] = 'komoditas' 
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Define the list of commodities
KOMODITAS_LIST = ['beras', 'telur', 'gula', 'daging', 'cabai']

# Gunakan ini saja (lebih baik karena portable)
MODEL_DIR = os.path.join('static', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

df_path = "harga_true.csv"

@app.template_filter('format_currency')
def format_currency(value):
    if isinstance(value, (int, float)):
        return f"Rp.{value:,.0f}"
    elif isinstance(value, str) and value.replace('.', '').isdigit():
        return f"Rp.{float(value):,.0f}"
    else:
        return f"Rp.{value}"  # Untuk nilai non-numerik

@app.template_filter('current_date')
def current_date(format='%Y-%m-%d'):
    return datetime.now().strftime(format)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'loggedin' in session:
        return redirect(url_for('train'))
    
    alert = None  # Kita akan menggunakan dictionary untuk alert
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Validasi client-side sudah dilakukan, ini backup validasi server-side
        if len(username) < 4 or len(password) < 6:
            alert = {
                'type': 'error',
                'message': 'Username harus minimal 4 karakter dan password minimal 6 karakter'
            }
        else:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM tb_user WHERE username = %s AND status = 'aktif'", (username,))
            account = cur.fetchone()
            cur.close()
            
            if account:
                if check_password_hash(account['password'], password):
                    session['loggedin'] = True
                    session['id'] = account['id']
                    session['username'] = account['username']
                    return redirect(url_for('train'))
                else:
                    alert = {
                        'type': 'error',
                        'message': 'Password yang Anda masukkan salah'
                    }
            else:
                alert = {
                    'type': 'error',
                    'message': 'Username tidak ditemukan atau akun tidak aktif'
                }
    
    return render_template('login.html', alert=alert)



@app.route('/users', methods=['GET', 'POST', 'PUT', 'DELETE'])
def users():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    alert = None
    cur = mysql.connection.cursor()
    if request.method == 'GET':
        cur.execute("SELECT id, username, status FROM tb_user ORDER BY id DESC")
        users = cur.fetchall()
        cur.close()
        return render_template('users.html', users=users)
    elif request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'create':
            username = request.form['username'].strip()
            password = request.form['password']
            
            if len(username) < 4:
                alert = {'type': 'error', 'message': 'Username must be at least 4 characters'}
            elif len(password) < 6:
                alert = {'type': 'error', 'message': 'Password must be at least 6 characters'}
            else:
                try:
                    cur.execute("SELECT id FROM tb_user WHERE username = %s", (username,))
                    if cur.fetchone():
                        alert = {'type': 'error', 'message': 'Username already exists'}
                    else:
                        hashed_pw = generate_password_hash(password)
                        cur.execute(
                            "INSERT INTO tb_user (username, password, status) VALUES (%s, %s, 'aktif')",
                            (username, hashed_pw)
                        )
                        mysql.connection.commit()
                        alert = {'type': 'success', 'message': 'User created successfully'}
                except Exception as e:
                    mysql.connection.rollback()
                    alert = {'type': 'error', 'message': f'Error: {str(e)}'}
                finally:
                    cur.close()
        
        elif action == 'update':
            # Update existing user
            user_id = request.form['user_id']
            username = request.form['username'].strip()
            password = request.form.get('password')
            status = request.form.get('status', 'aktif')  # Default to 'aktif' if not provided
            
            try:
                cur.execute("SELECT id FROM tb_user WHERE username = %s AND id != %s", 
                          (username, user_id))
                if cur.fetchone():
                    alert = {'type': 'error', 'message': 'Username already exists'}
                else:
                    if password:
                        if len(password) < 6:
                            alert = {'type': 'error', 'message': 'Password must be at least 6 characters'}
                        else:
                            hashed_pw = generate_password_hash(password)
                            cur.execute(
                                "UPDATE tb_user SET username = %s, password = %s, status = %s WHERE id = %s",
                                (username, hashed_pw, status, user_id)
                            )
                    else:
                        cur.execute(
                            "UPDATE tb_user SET username = %s, status = %s WHERE id = %s",
                            (username, status, user_id)
                        )
                    mysql.connection.commit()
                    alert = {'type': 'success', 'message': 'User updated successfully'}
            except Exception as e:
                mysql.connection.rollback()
                alert = {'type': 'error', 'message': f'Error: {str(e)}'}
            finally:
                cur.close()
        
        elif action == 'delete':
            user_id = request.form['user_id']         
            try:
                cur.execute("DELETE FROM tb_user WHERE id = %s", (user_id,))
                mysql.connection.commit()
                alert = {'type': 'success', 'message': 'User deleted successfully'}
            except Exception as e:
                mysql.connection.rollback()
                alert = {'type': 'error', 'message': f'Error: {str(e)}'}
            finally:
                cur.close()
        return redirect(url_for('users', alert=alert))
    
    elif request.method in ['PUT', 'DELETE']:
        try:
            data = request.get_json()
            user_id = data['user_id']
            
            if request.method == 'PUT':
                # API update
                username = data['username'].strip()
                password = data.get('password')
                status = data.get('status', 'aktif')
                
                if len(username) < 4:
                    return jsonify({'status': 'error', 'message': 'Username too short'}), 400
                
                if password and len(password) < 6:
                    return jsonify({'status': 'error', 'message': 'Password too short'}), 400
                
                cur.execute("SELECT id FROM tb_user WHERE username = %s AND id != %s", 
                          (username, user_id))
                if cur.fetchone():
                    return jsonify({'status': 'error', 'message': 'Username taken'}), 400
                
                if password:
                    hashed_pw = generate_password_hash(password)
                    cur.execute(
                        "UPDATE tb_user SET username = %s, password = %s, status = %s WHERE id = %s",
                        (username, hashed_pw, status, user_id)
                    )
                else:
                    cur.execute(
                        "UPDATE tb_user SET username = %s, status = %s WHERE id = %s",
                        (username, status, user_id)
                    )
                mysql.connection.commit()
                return jsonify({'status': 'success'})
            
            elif request.method == 'DELETE':
                # API delete
                cur.execute("DELETE FROM tb_user WHERE id = %s", (user_id,))
                mysql.connection.commit()
                return jsonify({'status': 'success'})
                
        except Exception as e:
            mysql.connection.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500
        finally:
            cur.close()
    
    return redirect(url_for('users'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    mape_scores = {}
    data_counts = {'total': 0,'training': 0,'testing': 0,'first_date': 0,'last_date': 0,}
    model_name = ""
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        if model_name:
            df = pd.read_csv(df_path)
            df['waktu'] = pd.to_datetime(df['waktu'])
            df = df.sort_values(by='waktu')

            first_date = df['waktu'].iloc[0]
            last_date = df['waktu'].iloc[-1]

            total_rows = len(df)
            train_size = int(total_rows * 0.8) 
            test_size = total_rows - train_size  
            
            data_counts = {'total': total_rows,'training': train_size,'testing': test_size,'first_date': first_date,'last_date': last_date,}

            df['day'] = df['waktu'].dt.day
            df['month'] = df['waktu'].dt.month
            df['year'] = df['waktu'].dt.year

            models = {}
            predictions_data = pd.DataFrame()
            scalers = {}
            lag_steps = 3

            for komoditas in KOMODITAS_LIST:
                # Create lag features
                temp_df = df[['waktu', 'day', 'month', 'year', komoditas]].copy()
                for lag in range(1, lag_steps + 1):
                    temp_df[f'lag{lag}'] = temp_df[komoditas].shift(lag)
                temp_df.dropna(inplace=True)
                
                features = ['day', 'month', 'year'] + [f'lag{i}' for i in range(1, lag_steps+1)]
                X = temp_df[features]
                y = temp_df[komoditas]
                
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
                
                model = lgb.LGBMRegressor(
                   boosting_type='gbdt', 
                   num_leaves=20, 
                   max_depth=-1, 
                   learning_rate=0.1, 
                   n_estimators=300
                )
                
                model.fit(X_train, y_train.ravel())
                
                # Predict and denormalize
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_test_actual = scaler_y.inverse_transform(y_test).flatten()
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
                mape_scores[komoditas] = round(mape, 2)
                
                # Store test data for visualization
                waktu_test = temp_df['waktu'].iloc[-len(y_test):]
                test_data = pd.DataFrame({
                    'waktu': waktu_test,
                    f'actual_{komoditas}': y_test_actual,
                    komoditas: y_pred
                }).tail(10)
                
                if predictions_data.empty:
                    predictions_data = test_data
                else:
                    predictions_data = predictions_data.merge(test_data, on='waktu', how='outer')
                
                # Save model and scalers
                models[komoditas] = {
                    'model': model,
                    'features': features,
                    'X_test': X_test,
                    'y_test': y_test_actual,
                    'y_pred': y_pred,
                    'last_date': df['waktu'].iloc[-1],
                    'scaler_X': scaler_X,
                    'scaler_y': scaler_y
                }
            # Save all models with additional date information
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            joblib.dump({
                'models': models,
                'mape_scores': mape_scores,
                'data_counts': data_counts,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'lag_steps': lag_steps,
                'predictions': predictions_data,
                'dataset_dates': {
                    'first_date': first_date,
                    'last_date': last_date
                }
            }, model_path)
    
    return render_template('train.html', 
                         mape_scores=mape_scores,
                         data_counts=data_counts,
                        )
    
def save_prediction(komoditas, selected_date, prediction):
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO prediksi (komoditas, tanggal, nilai_prediksi) VALUES (%s, %s, %s)",
            (komoditas, selected_date, prediction)
        )
        mysql.connection.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving prediction to MySQL: {e}")
        return False
    
    
@app.route('/', methods=['GET', 'POST'])
def predict():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    predictions = None
    selected_model = None
    selected_komoditas = None
    selected_date = None
    selected_date_formatted = None
    error = None  # Tambahkan variabel error

    if request.method == 'POST':
        selected_model = request.form.get('selected_model')
        selected_komoditas = request.form.get('selected_komoditas')
        selected_date = request.form.get('selected_date')
        
        if selected_model and selected_komoditas and selected_date:
            try:
                selected_date_dt = pd.to_datetime(selected_date)
                selected_date_formatted = selected_date_dt.strftime('%d-%m-%Y')
                
                model_path = os.path.join(MODEL_DIR, selected_model)
                model_data = joblib.load(model_path)
                
                komoditas_model = model_data['models'].get(selected_komoditas)
                if not komoditas_model:
                    error = "Model tidak ditemukan untuk komoditas ini"
                
                model = komoditas_model['model']
                features = komoditas_model['features']
                scaler_X = komoditas_model['scaler_X']
                scaler_y = komoditas_model['scaler_y']
                lag_steps = model_data.get('lag_steps', 3)
                
                # Baca data historis
                df_hist = pd.read_csv(df_path)
                df_hist['waktu'] = pd.to_datetime(df_hist['waktu'])
                df_komoditas = df_hist[['waktu', selected_komoditas]].copy()
            
                available_data = df_komoditas[selected_komoditas].tolist()
                predictions = []
                predicted_values = []  
                
                for day_offset in range(10): 
                    current_date = selected_date_dt + timedelta(days=day_offset)
                    input_data = {
                        'day': current_date.day,
                        'month': current_date.month,
                        'year': current_date.year
                    }
                    
                    lag_features = []
                    for lag in range(1, lag_steps + 1):
                        if len(available_data) >= lag:
                            lag_features.append(available_data[-lag])
                        elif len(predicted_values) >= (lag - len(available_data)):
                            lag_features.append(predicted_values[-(lag - len(available_data))])
                        else:
                            default_value = available_data[-1] if available_data else df_komoditas[selected_komoditas].mean()
                            lag_features.append(default_value)
                            
                    for i in range(1, lag_steps + 1):
                        input_data[f'lag{i}'] = lag_features[i-1]
                    input_df = pd.DataFrame([input_data])[features]
                    input_scaled = scaler_X.transform(input_df)
                    
                    pred_scaled = model.predict(input_scaled)[0]
                    
                    # Denormalisasi prediksi
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                    pred_rounded = round(float(pred), 2)
                    predictions.append(pred_rounded)
                    predicted_values.append(pred_rounded)  
                
                save_prediction(selected_komoditas, selected_date, predictions[0])
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return render_template('predict.html',
                                   model_files=model_files,
                                   komoditas_list=KOMODITAS_LIST,
                                   error=f"Terjadi kesalahan: {str(e)}")
    
    return render_template('predict.html', 
                         model_files=model_files, 
                         komoditas_list=KOMODITAS_LIST, 
                         predictions=predictions,
                         selected_date=selected_date,
                         selected_date_formatted=selected_date_formatted,
                         selected_model=selected_model, 
                         selected_komoditas=selected_komoditas,
                          error=error  )
    
    
@app.route('/coba', methods=['GET'])
def coba():
    return render_template('index.html')

@app.route('/records', methods=['GET', 'POST'])
def record():
    df = pd.read_csv("harga_true.csv", parse_dates=["waktu"])
    df['waktu'] = pd.to_datetime(df['waktu']).dt.date
    data = df.copy()
    start_date = request.form.get("start_date", "")
    end_date = request.form.get("end_date", "")

    # Validasi tanggal
    if start_date and end_date:
        try:
            start_date_dt = pd.to_datetime(start_date).date()
            end_date_dt = pd.to_datetime(end_date).date()
            if start_date_dt > end_date_dt:
                return render_template("record.html", error_message="Tanggal akhir tidak boleh lebih kecil dari tanggal awal.")
            data = data[(data["waktu"] >= start_date_dt) & (data["waktu"] <= end_date_dt)]
        except ValueError:
            return render_template("record.html", error_message="Format tanggal tidak valid.")

    # Buat plotly graph HTML untuk setiap komoditas tanpa modebar
    graphs = []
    commodities = ['beras', 'telur', 'gula', 'daging', 'cabai']
    for commodity in commodities:
        fig = px.line(
            data, 
            x="waktu", 
            y=commodity,
            title=f"Harga {commodity.capitalize()}",
            labels={"waktu": "Tanggal", commodity: "Harga (Rp)"}
        )
        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True)),
            template="plotly_white"
        )
        # Hilangkan modebar saat render HTML
        graph_html = fig.to_html(full_html=False, config={'displayModeBar': False})
        graphs.append(graph_html)

    return render_template(
        "record.html", 
        data=data.to_dict(orient="records"),
        start_date=start_date,
        end_date=end_date,
        graphs=graphs  # Kirim list graph HTML ke template
    )

@app.route('/history', methods=['GET'])
def history():
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT 
                id,
                komoditas,
                tanggal,
                nilai_prediksi,
                created_at
            FROM prediksi 
            ORDER BY created_at DESC
        """)
        all_data = cur.fetchall()
        cur.close()
        
        # Ambil hanya 10 data terbaru
        predictions_data = all_data[:10]
        
        return render_template("history.html", 
                            predictions_data=predictions_data,
                            error_message=None)
    except Exception as e:
        print(f"Error fetching prediction history: {e}")
        return render_template("history.html", 
                            predictions_data=None,
                            error_message="Gagal memuat riwayat prediksi.")
@app.route('/comment', methods=['POST'])
def comment():
    if request.method == 'POST':
        username = request.form.get('username')
        comment_text = request.form.get('comment')
        
        if username and comment_text:
            try:
                cur = mysql.connection.cursor()
                cur.execute(
                    "INSERT INTO feedback (username, comment) VALUES (%s, %s)",
                    (username, comment_text)
                )
                mysql.connection.commit()
                cur.close()
                return jsonify({"message": "Feedback berhasil dikirim!"})
            except Exception as e:
                print(f"Error saving feedback: {e}")
                return jsonify({"message": "Gagal mengirim feedback."}), 500
    
    return jsonify({"message": "Data feedback tidak lengkap."}), 400

@app.route('/akurasi', methods=['GET', 'POST'])
def akurasi():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    selected_model = None
    mape_scores = None
    chart_data = {} 
    data_counts = {'total': 0,'training': 0,'testing': 0,'first_date': 'N/A','last_date': 'N/A'}
    if request.method == 'POST':
        selected_model = request.form.get('selected_model')
        if selected_model:
            model_path = os.path.join(MODEL_DIR, selected_model)
            model_data = joblib.load(model_path)  # Muat file .pkl
            mape_scores = model_data.get('mape_scores', {})
            df_pred = model_data.get('predictions', pd.DataFrame())

            if 'data_counts' in model_data:
                data_counts.update({
                    'total': model_data['data_counts'].get('total', 0),
                    'training': model_data['data_counts'].get('training', 0),
                    'testing': model_data['data_counts'].get('testing', 0),
                    'first_date': model_data['data_counts'].get('first_date', 'N/A'),
                    'last_date': model_data['data_counts'].get('last_date', 'N/A')
                })
            
            # Format tanggal jika bukan string
            if not isinstance(data_counts['first_date'], str) and hasattr(data_counts['first_date'], 'strftime'):
                data_counts['first_date'] = data_counts['first_date'].strftime('%d-%m-%Y')
            if not isinstance(data_counts['last_date'], str) and hasattr(data_counts['last_date'], 'strftime'):
                data_counts['last_date'] = data_counts['last_date'].strftime('%d-%m-%Y')

            # Siapkan data untuk grafik
            if not df_pred.empty:
                chart_data = {
                    'labels': df_pred['waktu'].dt.strftime('%Y-%m-%d').tolist(),
                    'datasets': {}
                }

                for komoditas in KOMODITAS_LIST:
                    if f'actual_{komoditas}' in df_pred.columns:
                        chart_data['datasets'][komoditas] = {
                            'actual': df_pred[f'actual_{komoditas}'].tolist(),
                            'predicted': df_pred[komoditas].tolist()
                        }

    return render_template(
        'akurasi.html',
        model_files=model_files,
        selected_model=selected_model,
        mape_scores=mape_scores,
        chart_data=chart_data,
        data_counts=data_counts,  # Kirim data statistik ke template
        komoditas_list=KOMODITAS_LIST
    )

@app.route('/manage', methods=['GET', 'POST'])
def manage_data():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    status = request.args.get('status')
    message = request.args.get('message')
    
    try:
        cur = mysql.connection.cursor()
        
        # Get prediction data
        cur.execute("SELECT * FROM prediksi ORDER BY created_at DESC")
        data_prediksi = cur.fetchall()
        
        # Get feedback data
        cur.execute("SELECT * FROM feedback ORDER BY created_at DESC")
        data_feedback = cur.fetchall()
        
        cur.close()
        
        # Get list of saved models
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        model_info = []
        
        for model_file in model_files:
            model_path = os.path.join(MODEL_DIR, model_file)
            try:
                model_data = joblib.load(model_path)
                model_info.append({
                    'filename': model_file,
                    'training_date': model_data.get('training_date', 'Unknown'),
                    'mape_scores': model_data.get('mape_scores', {}),
                    'size': f"{os.path.getsize(model_path)/1024:.1f} KB",
                    'first_date': model_data.get('dataset_dates', {}).get('first_date', 'Unknown'),
                    'last_date': model_data.get('dataset_dates', {}).get('last_date', 'Unknown')
                })
            except Exception as e:
                print(f"Error loading model info for {model_file}: {e}")
                model_info.append({
                    'filename': model_file,
                    'error': 'Could not load model info'
                })

        # Handle delete request
        if request.method == 'POST':
            file_type = request.form.get('file_type')
            row_id = request.form.get('row_id')
            model_name = request.form.get('model_name')
            
            try:
                cur = mysql.connection.cursor()
                if file_type == 'prediction':
                    cur.execute("DELETE FROM prediksi WHERE id = %s", (row_id,))
                    mysql.connection.commit()
                    return redirect(url_for('manage_data', 
                                          status='success', 
                                          message='Data prediksi berhasil dihapus!'))
                elif file_type == 'feedback':
                    cur.execute("DELETE FROM feedback WHERE id = %s", (row_id,))
                    mysql.connection.commit()
                    return redirect(url_for('manage_data', 
                                          status='success', 
                                          message='Feedback berhasil dihapus!'))
                elif file_type == 'model':
                    model_path = os.path.join(MODEL_DIR, model_name)
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        return redirect(url_for('manage_data',
                                              status='success',
                                              message='Model berhasil dihapus!'))
                    else:
                        return redirect(url_for('manage_data',
                                              status='error',
                                              message='Model tidak ditemukan!'))
            except Exception as e:
                return redirect(url_for('manage_data', 
                                      status='error', 
                                      message=f'Error: {str(e)}'))
            finally:
                cur.close()
        
        return render_template('manage.html', 
                             data_prediksi=data_prediksi,
                             data_feedback=data_feedback,
                             model_info=model_info,
                             status=status,
                             message=message)
                             
    except Exception as e:
        print(f"Error in manage_data: {e}")
        return render_template('manage.html', 
                             error_message="Gagal memuat data dari database.")

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    try:
        # Define date range
        start_date = datetime.today().strftime('%m/%d/%Y')
        end_date = (datetime.today() - timedelta(days=10)).strftime('%m/%d/%Y')

        # Construct URL
        url = f"https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah?price_type_id=1&comcat_id=com_16%2Ccom_2%2Ccom_10%2Ccom_7%2Ccom_21&province_id=16&regency_id=47&market_id=&tipe_laporan=1&start_date={end_date}&end_date={start_date}&_=1747714827823"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        # Send request
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()['data']

            # Filter only level 2 commodities
            komoditas_detail = [item for item in data if item['level'] == 1]

            # Prepare data for HTML table
            table_data = []
            for item in komoditas_detail:
                nama = item['name']
                tanggal_harga = {k: v for k, v in item.items() if '/' in k}

                # Convert to list format
                rows = [[tanggal, harga.replace(",", "")] for tanggal, harga in tanggal_harga.items()]
                table_data.append({'nama': nama, 'rows': rows})

            # Debug: Print table_data
            print("Table Data:", table_data)

            return render_template('data_table.html', table_data=table_data)
        else:
            return f"Gagal mengambil data. Status code: {response.status_code}", 500
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", 500    
    


if __name__ == '__main__':
    app.run(debug=True)