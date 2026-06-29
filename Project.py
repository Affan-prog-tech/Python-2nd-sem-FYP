from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import json
import io
import base64

app = Flask(__name__)
app.secret_key = 'secret'

BASE_PATH = r"C:\Users\hp\Desktop\Project"
USER_FILE = f"{BASE_PATH}/user_finance_data.csv"
USERS_FILE = f"{BASE_PATH}/users.json"
os.makedirs(BASE_PATH, exist_ok=True)

def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(users):
    json.dump(users, open(USERS_FILE, 'w'))

def get_df():
    return pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame()

def save_df(df):
    df.to_csv(USER_FILE, index=False)

def fig64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def get_user_data(user):
    df = get_df()
    return df[df.username == user].to_dict('records') if not df.empty else []

def save_transaction(user, data):
    df = pd.concat([get_df(), pd.DataFrame([{'username': user, **data}])], ignore_index=True)
    save_df(df)

def delete_transaction(user, idx):
    df = get_df()
    if df.empty: return

    user_df = df[df.username == user]

    if idx < len(user_df):
        save_df(df.drop(user_df.iloc[idx].name))

def prepare_data(user):
    data = get_user_data(user)

    if len(data) < 3:
        return None

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = pd.to_numeric(df['amount'])

    return df

def split_data(df):
    inc = df[df.type == 'income']
    exp = df[df.type == 'expense']
    return inc, exp

def spending_analysis(df):
    if df is None or len(df) < 5: return None

    _, exp = split_data(df)

    if len(exp) < 3: return None

    exp['month'] = exp.date.dt.to_period('M')

    monthly = exp.groupby('month')['amount'].sum()

    top = exp.groupby('category')['amount'].sum().idxmax()

    text = [
        f"Highest category: {top}",
        f"Avg monthly: ${monthly.mean():.0f}"
    ]

    if len(monthly) > 1:
        change = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
        text.append(f"Spending {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}%")

    return " | ".join(text)

def predict_savings(df):
    if df is None or len(df) < 5: return None

    df['month_num'] = (df.date.dt.year - df.date.dt.year.min()) * 12 + df.date.dt.month

    monthly = df.groupby('month_num').apply(
        lambda x: x[x.type == 'income'].amount.sum() - x[x.type == 'expense'].amount.sum()
    ).reset_index(name='savings')

    if len(monthly) < 3: return None

    model = LinearRegression()
    model.fit(monthly[['month_num']], monthly['savings'])

    future = np.array([len(monthly) + i for i in range(1, 7)]).reshape(-1, 1)

    total = model.predict(future).sum()

    return f"Predicted savings: ${total:.0f}"

def predict_risk(df):
    if df is None or len(df) < 10: return None

    _, exp = split_data(df)

    if len(exp) < 5: return None

    exp['weekday'] = exp.date.dt.dayofweek
    exp['day'] = exp.date.dt.day

    avg = exp.groupby('category')['amount'].mean()

    exp['avg'] = exp.category.map(avg)
    exp['ratio'] = exp.amount / (exp.avg + 1)
    exp['over'] = (exp.amount > exp.avg * 1.5).astype(int)

    if exp.over.nunique() < 2: return None

    X = exp[['amount', 'weekday', 'day', 'ratio']]
    y = exp['over']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    scaler = StandardScaler()

    model = LogisticRegression(max_iter=1000)

    model.fit(scaler.fit_transform(X_train), y_train)

    risk = model.predict_proba(
        scaler.transform(X.mean().values.reshape(1, -1))
    )[0][1]

    level = "HIGH" if risk > .6 else "MEDIUM" if risk > .3 else "LOW"

    return f"{level} RISK: {risk * 100:.0f}% chance of overspending"

def get_advice(df):
    if df is None or len(df) < 5: return None

    inc, exp = split_data(df)

    income, expense = inc.amount.sum(), exp.amount.sum()

    if income == 0: return None

    rate = (income - expense) / income * 100

    advice = [
        f"Savings rate: {rate:.0f}%"
        if rate < 20 else
        f"Excellent savings rate: {rate:.0f}%"
    ]

    top = exp.groupby('category')['amount'].sum()

    if len(top):
        advice.append(
            f"Reduce {top.idxmax()} spending"
        )

    advice.append(f"Save ${(income * .2) / 30:.0f}/day")

    return " | ".join(advice)

def create_3d_plot(df):
    if df is None or len(df) < 5:
        return None

    df['month'] = df.date.dt.to_period('M').astype(str)
    inc, exp = split_data(df)

    monthly_income = inc.groupby('month')['amount'].sum()
    monthly_expense = exp.groupby('month')['amount'].sum()

    months = sorted(set(monthly_income.index) | set(monthly_expense.index))

    if len(months) < 2:
        return None

    incomes = [monthly_income.get(m, 0) for m in months]
    expenses = [monthly_expense.get(m, 0) for m in months]
    savings = [i - e for i, e in zip(incomes, expenses)]

    month_indices = list(range(len(months)))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(month_indices, [0] * len(months), incomes, c='green', marker='o', s=100, label='Income')
    ax.scatter(month_indices, [1] * len(months), expenses, c='red', marker='s', s=100, label='Expense')
    ax.scatter(month_indices, [2] * len(months), savings, c='blue', marker='^', s=100, label='Savings')

    ax.set_xlabel('Month')
    ax.set_ylabel('Type (0=Income,1=Expense,2=Savings)')
    ax.set_zlabel('Amount ($)')
    ax.set_title('3D Financial Overview')
    ax.legend()

    ax.set_xticks(month_indices)
    ax.set_xticklabels(months, rotation=45, ha='right')

    ax.view_init(elev=20, azim=45)

    return fig64(fig)

def create_charts(df, budget):
    if df is None or len(df) < 3: return []

    charts = []

    inc, exp = split_data(df)

    exp['month'] = exp.date.dt.to_period('M').astype(str)
    monthly = exp.groupby('month')['amount'].sum()

    if len(exp):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly.index, monthly.values, marker='o')
        ax.set_title('Monthly Spending')
        plt.xticks(rotation=45)
        charts.append(fig64(fig))

    if len(exp):
        fig, ax = plt.subplots(figsize=(7, 5))
        exp.groupby('category')['amount'].sum().plot(kind='pie', autopct='%1.0f%%', ax=ax)
        ax.set_title('Category Distribution')
        charts.append(fig64(fig))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(['Income', 'Expense'], [inc.amount.sum(), exp.amount.sum()])
    ax.set_title('Income vs Expense')
    charts.append(fig64(fig))

    if len(df) > 5:
        fig, ax = plt.subplots(figsize=(10, 5))

        df['month'] = df.date.dt.to_period('M').astype(str)

        savings = df.groupby('month').apply(
            lambda x: x[x.type == 'income'].amount.sum() - x[x.type == 'expense'].amount.sum()
        )

        ax.bar(savings.index, savings.values)
        ax.axhline(0)
        ax.set_title('Monthly Savings')
        plt.xticks(rotation=45)

        charts.append(fig64(fig))

    if len(exp) > 5:
        fig, ax = plt.subplots(figsize=(10, 5))

        exp['weekday'] = exp.date.dt.day_name()
        exp['week'] = exp.date.dt.isocalendar().week

        heat = exp.pivot_table(
            values='amount',
            index='week',
            columns='weekday',
            aggfunc='sum',
            fill_value=0
        )

        sns.heatmap(heat, cmap='YlOrRd', ax=ax)

        ax.set_title('Spending Heatmap')

        charts.append(fig64(fig))

    if budget:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            ['Spending', 'Budget'],
            [exp.amount.sum(), budget]
        )

        ax.set_title('Budget Comparison')

        charts.append(fig64(fig))

    return charts

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        u, p, c = request.form['username'], request.form['password'], request.form['confirm_password']

        users = load_users()

        if u in users:
            return render_template('register.html', error='Username exists')

        if p != c:
            return render_template('register.html', error="Passwords don't match")

        if len(p) < 4:
            return render_template('register.html', error='Password too short')

        users[u] = p
        save_users(users)

        return redirect(url_for('login'))

    return render_template('register.html', error=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        u, p = request.form['username'], request.form['password']

        users = load_users()

        if u in users and users[u] == p:
            session['username'] = u
            return redirect(url_for('dashboard'))

        return render_template('login.html', error='Invalid credentials')

    return render_template('login.html', error=None)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    if 'username' in session:
        save_transaction(session['username'], {
            'date': request.form['date'],
            'type': request.form['type'],
            'category': request.form['category'],
            'amount': float(request.form['amount'])
        })

    return redirect(url_for('dashboard'))

@app.route('/delete/<int:idx>')
def delete(idx):
    if 'username' in session:
        delete_transaction(session['username'], idx)

    return redirect(url_for('dashboard'))

@app.route('/budget', methods=['POST'])
def budget():
    if 'username' in session:
        session['budget_target'] = float(request.form['budget'])

    return redirect(url_for('dashboard'))

@app.route('/')
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    user = session['username']

    df = prepare_data(user)

    budget = session.get('budget_target', 2000)

    charts = create_charts(df, budget)

    return render_template(
        'dashboard.html',
        username=user,
        transactions=get_user_data(user)[-15:][::-1],
        charts=charts,
        three_d_chart=create_3d_plot(df),
        prediction=predict_savings(df),
        alert=predict_risk(df),
        advice=get_advice(df),
        pattern=spending_analysis(df),
        budget=budget,
        budget_chart=charts[-1] if charts else None,
        today=datetime.now().strftime('%Y-%m-%d')
    )

if __name__ == '__main__':

    if not os.path.exists(USERS_FILE):
        save_users({})

    users = load_users()

    print("""
Finance Tracker Running
http://127.0.0.1:5000
""")

    app.run(debug=True)