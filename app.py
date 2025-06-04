from flask import Flask, render_template, request, redirect, url_for, flash, session
import json
import numpy as np
import psycopg2
from psycopg2 import sql
from flask import make_response, render_template, session, flash, redirect, url_for
from xhtml2pdf import pisa
import io
import pdfkit
from flask import make_response
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
import json
import numpy as np
import psycopg2
from psycopg2 import sql
import pdfkit
import requests
import os
import psycopg2
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Thay bằng secret key thật của bạn
OPENROUTER_API_KEY = "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"

# -------------------------------
# DATABASE: Kết nối đến PostgreSQL
# -------------------------------
# def get_db_connection():
#     conn = psycopg2.connect(
#         database='ahp_tourism_db',  # Đổi tên database
#         user='postgres',
#         password='oikuhj12345',
#         host='localhost',
#         port='5432'
#     )
#     return conn
def get_db_connection():
    db_url = os.environ.get('DATABASE_URL')

    if db_url:
        # Xử lý chuỗi URL do Render cung cấp
        up.uses_netloc.append("postgres")
        url = up.urlparse(db_url)
        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
    else:
        # Cấu hình chạy local
        conn = psycopg2.connect(
            database='ahp_tourism_db',
            user='postgres',
            password='oikuhj12345',
            host='localhost',
            port='5432'
        )

    return conn

# -------------------------------
# Truy vấn cấu hình tiêu chí từ bảng criteria_config
# -------------------------------
def get_criteria_config():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT key, display, table_name, field_name, is_cost FROM criteria_config ORDER BY id;")
        rows = cur.fetchall()
        conn.close()
        config = { row[0]: {'display': row[1], 'table': row[2], 'field': row[3], 'is_cost': row[4]} for row in rows }
        return config
    except Exception as e:
        flash("Lỗi truy xuất dữ liệu cấu hình tiêu chí: " + str(e))
        return {}

# -------------------------------
# Hàm tính AHP
# -------------------------------
def compute_ahp(matrix):
    matrix = np.array(matrix, dtype=float)
    col_sum = matrix.sum(axis=0)
    norm_matrix = matrix / col_sum
    w = norm_matrix.mean(axis=1)
    weighted_sum = np.dot(matrix, w)
    lambda_max = (weighted_sum / w).mean()
    n = len(w)
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1:0,2:0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32}
    RI = RI_dict.get(n,1.32)
    CR = CI / RI if RI != 0 else 0
    return w, lambda_max, CI, CR

def validate_value(val):
    return val > 0

def save_calculation_history(destinations, criteria, crit_weights, results, matrices):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO calculation_history (destinations, criteria, crit_weights, results, matrices)
            VALUES (%s, %s, %s, %s, %s);
        """
        cur.execute(query, (
            json.dumps(destinations),
            json.dumps(criteria),
            json.dumps(crit_weights),
            json.dumps(results),
            json.dumps(matrices)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        flash("Lỗi lưu lịch sử tính toán: " + str(e))

# -------------------------------
# ROUTE: Trang chủ -> lọc địa điểm
# -------------------------------
@app.route('/')
def index():
    return redirect(url_for('filter_destinations'))

# -------------------------------
# ROUTE: Lọc điểm du lịch theo vùng, loại hình
# -------------------------------
@app.route('/filter', methods=['GET', 'POST'])
def filter_destinations():
    if request.method == 'POST':
        selected_regions = request.form.getlist('region')
        selected_types = request.form.getlist('type')
        if not selected_regions:
            flash("Vui lòng chọn ít nhất 1 vùng.")
            return redirect(url_for('filter_destinations'))
        if not selected_types:
            flash("Vui lòng chọn ít nhất 1 loại hình.")
            return redirect(url_for('filter_destinations'))
        session['regions'] = selected_regions
        session['types'] = selected_types
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            query = sql.SQL("""
                SELECT name FROM destinations
                WHERE region = ANY(%s) AND type = ANY(%s)
                ORDER BY name;
            """)
            cur.execute(query, (selected_regions, selected_types))
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            flash("Lỗi truy xuất dữ liệu điểm du lịch: " + str(e))
            return redirect(url_for('filter_destinations'))
        if not rows:
            flash("Không có điểm du lịch thỏa mãn.")
            return redirect(url_for('filter_destinations'))
        dests = [r[0] for r in rows]
        session['destinations'] = dests
        return render_template('select_destinations.html', destinations=dests)
    else:
        # ==== LOAD danh mục vùng / loại ====
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT region FROM destinations ORDER BY region;")
            regions = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT type FROM destinations ORDER BY type;")
            types = [t[0] for t in cur.fetchall()]
            cur.execute("SELECT region, type FROM destinations;")
            region_types = {}
            for region, typ in cur.fetchall():
                region_types.setdefault(region, set()).add(typ)
            region_types = {r: sorted(list(ts)) for r, ts in region_types.items()}
            conn.close()
        except Exception as e:
            flash("Lỗi truy xuất danh mục vùng/loại: " + str(e))
            regions, types, region_types = [], [], {}

        # ==== TÍNH TOP 5 ĐIỂM ĐƯỢC CHỌN NHIỀU NHẤT ====
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT destinations FROM calculation_history;")
            rows = cur.fetchall()
            conn.close()
            counts = {}
            for (dest_list_json,) in rows:
                dests = json.loads(dest_list_json)
                for d in dests:
                    counts[d] = counts.get(d, 0) + 1
            top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_labels = [t[0] for t in top5]
            top5_counts = [t[1] for t in top5]
        except Exception as e:
            top5_labels, top5_counts = [], []

        return render_template(
            'filter.html',
            regions=regions,
            types=types,
            region_types=region_types,
            top5_labels=top5_labels,
            top5_counts=top5_counts
        )


# -------------------------------
# ROUTE: Chọn tất cả điểm du lịch
# -------------------------------
@app.route('/select_all')
def select_all_destinations():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM destinations ORDER BY name;")
        dests = [r[0] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        flash("Lỗi truy xuất điểm du lịch: " + str(e))
        return redirect(url_for('filter_destinations'))
    session['destinations'] = dests
    return render_template('select_destinations.html', destinations=dests)

# -------------------------------
# ROUTE: Lưu điểm đã chọn
# -------------------------------
@app.route('/select', methods=['POST'])
def select_destinations():
    selected = request.form.getlist('destination')
    if not selected:
        flash("Vui lòng chọn ít nhất 1 điểm.")
        return redirect(url_for('filter_destinations'))
    session['selected_destinations'] = selected
    flash("Đã lưu điểm du lịch. Tiếp theo, chọn tiêu chí AHP (2–7).")
    return redirect(url_for('select_criteria_page'))

# -------------------------------
# ROUTE: Chọn tiêu chí
# -------------------------------
@app.route('/select_criteria', methods=['GET'])
def select_criteria_page():
    config = get_criteria_config()
    options = [{'value':k, 'display':config[k]['display']} for k in config]
    return render_template('select_criteria.html', criteria_options=options)

@app.route('/criteria', methods=['POST'])
def save_criteria():
    sel = request.form.getlist('criteria')
    if len(sel)<2 or len(sel)>7:
        flash("Vui lòng chọn từ 2 đến 7 tiêu chí.")
        return redirect(url_for('select_criteria_page'))
    session['selected_criteria'] = sel
    return render_template('criteria.html', crits=sel)

@app.route('/criteria_matrix', methods=['POST'])
def criteria_matrix():
    selected = session.get('selected_criteria',[])
    n = len(selected)
    try:
        matrix = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                v = float(request.form[f"cell_{i}_{j}"])
                if not validate_value(v): raise ValueError
                matrix[i][j]=v; matrix[j][i]=1/v
    except:
        flash("Giá trị ma trận phải > 0 và hợp lệ.")
        return redirect(url_for('save_criteria'))
    w, lm, CI, CR = compute_ahp(matrix)
    if CR>=0.1:
        flash(f"Ma trận tiêu chí không nhất quán (CR={CR:.3f}).")
        return redirect(url_for('save_criteria'))
    session['crit_weights']=w.tolist()
    session['criteria_consistency']={'lambda_max':lm,'CI':CI,'CR':CR}
    flash("Đã lưu trọng số tiêu chí.")
    return redirect(url_for('result'))

# @app.route('/result')
# def result():
#     selected = session.get('selected_criteria',[])
#     crit_weights = session.get('crit_weights',[])
#     if not selected or not crit_weights:
#         flash("Thiếu tiêu chí hoặc trọng số.")
#         return redirect(url_for('select_criteria_page'))
#     full_cfg = get_criteria_config()
#     chosen = session.get('selected_destinations',[])
#     sub_vectors={}
#     names=None
#     # xây ma trận từng tiêu chí
#     for crit in selected:
#         info = full_cfg[crit]
#         try:
#             conn=get_db_connection(); cur=conn.cursor()
#             q=sql.SQL("SELECT dest_name AS name, {field} FROM {table} WHERE dest_name=ANY(%s) ORDER BY dest_name;").format(
#                 field=sql.Identifier(info['field']),
#                 table=sql.Identifier(info['table'])
#             )
#             cur.execute(q,(chosen,)); rows=cur.fetchall(); conn.close()
#         except Exception as e:
#             flash("Lỗi dữ liệu "+str(e)); return redirect(url_for('select_criteria_page'))
#         if names is None: names=[r[0] for r in rows]
#         vals=[float(r[1]) for r in rows]
#         m=len(vals)
#         M=[[1 if i==j else 0 for j in range(m)] for i in range(m)]
#         for i in range(m):
#             for j in range(i+1,m):
#                 if info['is_cost']:
#                     ratio = vals[j]/vals[i]
#                 else:
#                     ratio = vals[i]/vals[j]
#                 M[i][j]=ratio; M[j][i]=1/ratio
#         lw,_,_,lCR=compute_ahp(M)
#         if lCR>=0.1: flash(f"Ma trận {info['display']} không nhất quán"); return redirect(url_for('select_criteria_page'))
#         sub_vectors[crit]={ 'original':M, 'weights':lw.tolist() }
#     # tính điểm tổng hợp
#     results=[]
#     for i,name in enumerate(names):
#         score= sum(crit_weights[idx]*sub_vectors[crit]['weights'][i] for idx,crit in enumerate(selected))
#         results.append({'name':name,'score':score})
#     # chuẩn hóa
#     total=sum(r['score'] for r in results)
#     if total>0:
#         for r in results: r['score']/=total
#     results=sorted(results,key=lambda x: x['score'],reverse=True)
#     session['matrices_detail']=sub_vectors
#     session['alternative_names']=names
#     save_calculation_history(chosen, selected, crit_weights, results, sub_vectors)
#     c=session.get('criteria_consistency',{})
#     return render_template('result.html',
#         results=results,
#         crit_weights=crit_weights,
#         selected_criteria=session.get('selected_criteria', []),
#         lambda_max=c.get('lambda_max'),
#         ci=c.get('CI'),
#         cr=c.get('CR')
#     )
@app.route('/result')
def result():
    selected = session.get('selected_criteria',[])
    crit_weights = session.get('crit_weights',[])
    if not selected or not crit_weights:
        flash("Thiếu tiêu chí hoặc trọng số.")
        return redirect(url_for('select_criteria_page'))
    full_cfg = get_criteria_config()
    chosen = session.get('selected_destinations',[])
    sub_vectors={}
    names=None
    for crit in selected:
        info = full_cfg[crit]
        try:
            conn=get_db_connection(); cur=conn.cursor()
            q=sql.SQL("SELECT dest_name AS name, {field} FROM {table} WHERE dest_name=ANY(%s) ORDER BY dest_name;").format(
                field=sql.Identifier(info['field']),
                table=sql.Identifier(info['table'])
            )
            cur.execute(q,(chosen,)); rows=cur.fetchall(); conn.close()
        except Exception as e:
            flash("Lỗi dữ liệu "+str(e)); return redirect(url_for('select_criteria_page'))
        if names is None: names=[r[0] for r in rows]
        vals=[float(r[1]) for r in rows]
        m=len(vals)
        M=[[1 if i==j else 0 for j in range(m)] for i in range(m)]
        for i in range(m):
            for j in range(i+1,m):
                if info['is_cost']:
                    ratio = vals[j]/vals[i]
                else:
                    ratio = vals[i]/vals[j]
                M[i][j]=ratio; M[j][i]=1/ratio
        lw,_,_,lCR=compute_ahp(M)
        if lCR>=0.1: flash(f"Ma trận {info['display']} không nhất quán"); return redirect(url_for('select_criteria_page'))
        sub_vectors[crit]={ 'original':M, 'weights':lw.tolist() }
    results=[]
    for i,name in enumerate(names):
        score= sum(crit_weights[idx]*sub_vectors[crit]['weights'][i] for idx,crit in enumerate(selected))
        results.append({'name':name,'score':score})
    total=sum(r['score'] for r in results)
    if total>0:
        for r in results: r['score']/=total
    results=sorted(results,key=lambda x: x['score'],reverse=True)
    session['matrices_detail']=sub_vectors
    session['alternative_names']=names
    session['results']=results  # Lưu lại để xuất PDF
    c={'lambda_max':None,'CI':None,'CR':None}
    # Tính lại consistency, nếu đã có session thì lấy từ đó
    consistency = session.get('criteria_consistency',{})
    c['lambda_max'] = consistency.get('lambda_max')
    c['CI'] = consistency.get('CI')
    c['CR'] = consistency.get('CR')
    save_calculation_history(chosen, selected, crit_weights, results, sub_vectors)
    return render_template('result.html',
        results=results,
        crit_weights=crit_weights,
        selected_criteria=selected,
        lambda_max=c.get('lambda_max'),
        ci=c.get('CI'),
        cr=c.get('CR'),
        pdf_mode=False
    )

    
@app.route('/ma-tran', methods=['GET','POST'])
def matrix_display():
    matrices = session.get('matrices_detail',{})
    names = session.get('alternative_names',[])
    return render_template('matrix_display.html', matrices=matrices, alternatives=names)

@app.route('/custom_matrix', methods=['POST'])
def custom_matrix():
    # Lấy các biến cũ
    selected = session.get('selected_criteria', [])
    crit_weights = session.get('crit_weights', [])
    chosen = session.get('selected_destinations', [])
    full_cfg = get_criteria_config()

    # Đọc ma trận con do user nhập
    sub_vectors = {}
    for crit in selected:
        info = full_cfg[crit]
        # kích thước ma trận
        n = len(chosen)
        # khởi ma trận 2D
        M = [[1.0]*n for _ in range(n)]
        # điền tam giác trên từ form
        for i in range(n):
            for j in range(i+1, n):
                key = f"matrix_{crit}_{i}_{j}"
                try:
                    v = float(request.form[key])
                    if v<=0: raise ValueError
                except:
                    flash(f"Giá trị ma trận {crit} không hợp lệ.")
                    return redirect(url_for('matrix_display'))
                M[i][j] = v
                M[j][i] = 1.0 / v
        # tính AHP cho ma trận này
        lw, _, _, lCR = compute_ahp(M)
        if lCR >= 0.1:
            flash(f"Ma trận con cho {info['display']} không nhất quán (CR={lCR:.3f})")
            return redirect(url_for('matrix_display'))
        sub_vectors[crit] = {'original': M, 'weights': lw.tolist()}

    # Tính kết quả tổng hợp
    names = chosen
    results = []
    for idx, name in enumerate(names):
        score = sum( crit_weights[k]*sub_vectors[crit]['weights'][idx]
                     for k,crit in enumerate(selected) )
        results.append({'name': name, 'score': score})
    total = sum(r['score'] for r in results)
    if total>0:
        for r in results: r['score'] /= total
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Cập nhật session và lưu lịch sử
    session['matrices_detail'] = sub_vectors
    session['alternative_names'] = names
    save_calculation_history(names, selected, crit_weights, results, sub_vectors)

    # Trả về trang result.html như trước
    c = session.get('criteria_consistency', {})
    return render_template('result.html',
        results=results,
        crit_weights=crit_weights,
        selected_criteria=selected,
        lambda_max=c.get('lambda_max'),
        ci=c.get('CI'),
        cr=c.get('CR')
    )

@app.route('/history')
def history():
    try:
        conn=get_db_connection(); cur=conn.cursor()
        cur.execute("SELECT id, calc_time, destinations, criteria, crit_weights, results FROM calculation_history ORDER BY calc_time DESC;")
        rows=cur.fetchall(); conn.close()
        history_list=[{
            'id':r[0],'calc_time':r[1],
            'destinations':json.loads(r[2]),
            'criteria':json.loads(r[3]),
            'crit_weights':json.loads(r[4]) if r[4] else None,
            'results':json.loads(r[5]) if r[5] else None
        } for r in rows]
    except:
        history_list=[]
    return render_template('history.html', history_list=history_list)

# @app.route('/export', methods=['GET'])
# def export_pdf():
#     selected_criteria = session.get('selected_criteria')
#     crit_weights = session.get('crit_weights')
#     lambda_max = session.get('lambda_max')
#     ci = session.get('ci')
#     cr = session.get('cr')
#     results = session.get('results')

#     if not all([selected_criteria, crit_weights, lambda_max, ci, cr, results]):
#         flash("Dữ liệu không đầy đủ để xuất PDF.")
#         return redirect(url_for('result'))

#     rendered = render_template(
#         'report.html',
#         selected_criteria=selected_criteria,
#         crit_weights=crit_weights,
#         lambda_max=lambda_max,
#         ci=ci,
#         cr=cr,
#         results=results
#     )

#     pdf = pdfkit.from_string(rendered, False)
#     response = make_response(pdf)
#     response.headers['Content-Type'] = 'application/pdf'
#     response.headers['Content-Disposition'] = 'attachment; filename=ahp_report.pdf'
#     return response
@app.route('/export', methods=['GET'])
def export_pdf():
    selected_criteria = session.get('selected_criteria')
    crit_weights = session.get('crit_weights')
    lambda_max = session.get('criteria_consistency',{}).get('lambda_max')
    ci = session.get('criteria_consistency',{}).get('CI')
    cr = session.get('criteria_consistency',{}).get('CR')
    results = session.get('results')

    if not all([selected_criteria, crit_weights, lambda_max is not None, ci is not None, cr is not None, results]):
        flash("Dữ liệu không đầy đủ để xuất PDF.")
        return redirect(url_for('result'))

    rendered = render_template(
        'report.html',
        selected_criteria=selected_criteria,
        crit_weights=crit_weights,
        lambda_max=lambda_max,
        ci=ci,
        cr=cr,
        results=results
    )

    pdf = pdfkit.from_string(rendered, False)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=ahp_report.pdf'
    return response

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        if not user_input:
            flash("Vui lòng nhập nội dung.")
            return redirect(url_for('chat'))

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openai/gpt-3.5-turbo",  # bạn có thể chọn model khác
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý du lịch."},
                {"role": "user", "content": user_input}
            ]
        }

        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers, data=json.dumps(data))
            res.raise_for_status()
            ai_message = res.json()['choices'][0]['message']['content']
        except Exception as e:
            flash(f"Lỗi API: {e}")
            ai_message = "Xin lỗi, đã xảy ra lỗi khi gọi AI."

        return render_template("chat.html", user_input=user_input, ai_response=ai_message)

    return render_template("chat.html")


if __name__ == '__main__':
    app.run(debug=True)
