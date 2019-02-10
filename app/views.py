from flask import render_template, redirect, url_for, request

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, IntegerField, FloatField
from wtforms.validators import InputRequired, Email, Length, NumberRange
from app import app

from algorithm import clf
from algorithm2 import bdt


import pandas as pd
######################################################################################

class PredictionForm(FlaskForm):

	age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=1, max=120)])
	male = IntegerField('Male', validators=[InputRequired(), NumberRange(min=0, max=1)])
	scholarship = IntegerField('Scholarship', validators=[InputRequired(), NumberRange(min=0, max=1)])
	diabetes = IntegerField('Diabetes', validators=[InputRequired(), NumberRange(min=0, max=1)])
	alcoholism = IntegerField('Alcoholism', validators=[InputRequired(), NumberRange(min=0, max=1)])
	handcap = IntegerField('No. of Handicap', validators=[InputRequired(), NumberRange(min=0, max=8)])
	hipertension = IntegerField('Hipertension', validators=[InputRequired(), NumberRange(min=0, max=1)])

class PredictionLifeStyleForm(FlaskForm):

	male = IntegerField('Male', validators=[InputRequired(), NumberRange(min=0, max=1)])
	marital_status = IntegerField('Marital Status', validators=[InputRequired(), NumberRange(min=0, max=1)])
	children = IntegerField('Number of Children', validators=[InputRequired(), NumberRange(min=0, max=8)])
	avg_commute = FloatField('Average Commute Time in Hrs', validators=[InputRequired(), NumberRange(min=0, max=9)])

	daily_internet_use = FloatField('Daily Interet Use in Hrs', validators=[InputRequired(), NumberRange(min=0, max=16)])
	available_vehicles = IntegerField('Number of Available Vehicles', validators=[InputRequired(), NumberRange(min=0, max=10)])
	military_service = IntegerField('Military Service Done or Not', validators=[InputRequired(), NumberRange(min=0, max=1)])
	employed = IntegerField('Employement Status 1 or 0', validators=[InputRequired(), NumberRange(min=0, max=1)])

	retired = IntegerField('Retirement Status 1 or 0', validators=[InputRequired(), NumberRange(min=0, max=1)])
	student = IntegerField('Student Status 1 or 0', validators=[InputRequired(), NumberRange(min=0, max=1)])
	education_bachelors = IntegerField('Bachelors Status 1 or 0', validators=[InputRequired(), NumberRange(min=0, max=1)])
	education_highschool = IntegerField('HighSchool Status 1 or 0', validators=[InputRequired(), NumberRange(min=0, max=1)])


####################################################################################################################################

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

@app.route('/dashboard_lifestyle')
def dashboard_lifestyle():
	d1 = request.args.get('d1')
	d2 = request.args.get('d2')
	d3 = request.args.get('d3')
	d4 = request.args.get('d4')
	d5 = request.args.get('d5')
	d6 = request.args.get('d6')
	d7 = request.args.get('d7')
	d8 = request.args.get('d8')
	d9 = request.args.get('d9')
	d10 = request.args.get('d10')

	most_likely_disease = request.args.get('most_likely_disease')

	return render_template('dashboard_lifestyle.html', d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6, d7=d7,
							d8=d8, d9=d9, d10=d10, most_likely_disease=most_likely_disease)
	
#####################################################################################################################################

### Algorithm 1 ###
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

		show_prob = pred_proba[0]
		noshow_prob = pred_proba[1]

		return redirect(url_for('dashboard', show_prob = show_prob, noshow_prob = noshow_prob))

	return render_template('predict.html', form=form)

####################################################################################################################################

### Algorithm 2 ###
@app.route('/predict_lifestyle', methods=['POST', 'GET'])
def predict_lifestyle():
	form = PredictionLifeStyleForm()
	if form.validate_on_submit():
	
		male = int(form.male.data)
		marital_status = int(form.marital_status.data)
		children = int(form.children.data)
		avg_commute = float(form.avg_commute.data)

		daily_internet_use = float(form.daily_internet_use.data)
		available_vehicles = int(form.available_vehicles.data)
		military_service = int(form.military_service.data)
		employed = int(form.employed.data)
		
		retired = int(form.retired.data)
		student = int(form.student.data)
		education_bachelors = int(form.education_bachelors.data)
		education_highschool = int(form.education_highschool.data)


		### Make Pediction with clf
		patient0 = [[
			male,
			marital_status,
			children,
			avg_commute,

			daily_internet_use,
			available_vehicles,
			military_service,
			employed,

			retired,
			student,
			education_bachelors,
			education_highschool
		]]


		pred_lifestyle = pd.DataFrame(bdt.predict_proba(patient0), columns=bdt.classes_)

		disease_probs = pred_lifestyle.values.tolist()[0]
		print("Disease_Probs, ", disease_probs)

		d1 = disease_probs[0]
		d2 = disease_probs[1]
		d3 = disease_probs[2]
		d4 = disease_probs[3]
		d5 = disease_probs[4]
		d6 = disease_probs[5]
		d7 = disease_probs[6]
		d8 = disease_probs[7]
		d9 = disease_probs[8]
		d10 = disease_probs[9]
		
		most_likely_disease = bdt.predict(patient0)[0]


		return redirect(url_for('dashboard_lifestyle', most_likely_disease = most_likely_disease,
			 d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6, d7=d7, d8=d8, d9=d9, d10=d10))

	return render_template('predict_lifestyle.html', form=form)


#####################################################################################################################################

