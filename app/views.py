from flask import render_template, redirect, url_for, request

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, IntegerField, FloatField
from wtforms.validators import InputRequired, Email, Length, NumberRange
from app import app

from algorithm import clf

class PredictionForm(FlaskForm):

	age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=1, max=120)])
	male = IntegerField('Male', validators=[InputRequired(), NumberRange(min=0, max=1)])
	scholarship = IntegerField('Scholarship', validators=[InputRequired(), NumberRange(min=0, max=1)])
	diabetes = IntegerField('Diabetes', validators=[InputRequired(), NumberRange(min=0, max=1)])
	alcoholism = IntegerField('Alcoholism', validators=[InputRequired(), NumberRange(min=0, max=1)])
	handcap = IntegerField('No. of Handicap', validators=[InputRequired(), NumberRange(min=0, max=8)])
	hipertension = IntegerField('Hipertension', validators=[InputRequired(), NumberRange(min=0, max=1)])


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/dashboard')
def dashboard():
	show_prob = request.args.get('show_prob')
	noshow_prob = request.args.get('noshow_prob')
	return render_template('dashboard.html', show_prob=show_prob, noshow_prob=noshow_prob)
	

@app.route('/predict', methods=['POST', 'GET'])
def predict():
	form = PredictionForm()
	if form.validate_on_submit():
	
		age = int(form.age.data)
		male = int(form.male.data)
		scholarship = int(form.scholarship.data)
		diabetes = int(form.diabetes.data)
		alcoholism = int(form.alcoholism.data)
		handcap = int(form.handcap.data)
		hipertension = int(form.hipertension.data)


		### Make Pediction with clf
		patient0 = [[
			age,
			male,
			scholarship,
			diabetes,
			alcoholism,
			handcap,
			hipertension
		
		]]


		# pred = clf.predict(patient0)[0]
		pred_proba = clf.predict_proba(patient0)[0]

		show_prob = pred_proba[0] * 100.0
		show_prob = round(show_prob, 2)
		noshow_prob = pred_proba[1] * 100.0
		noshow_prob = round(noshow_prob, 2)

		return redirect(url_for('dashboard', show_prob = show_prob, noshow_prob = noshow_prob))

	return render_template('predict.html', form=form)
