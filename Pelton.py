import numpy as np
import matplotlib.pyplot as plt

g = 9.81 #gravitational acceleration
gamma = 9810 #rho * g specific weight of water
rho = 1000 #density of water @ 25deg Celsius
mu = 1e-3 #dynamic viscosity of Water @25deg Celsius

materials = ('Wooden Stave Pipe','Fiberglass Pipe','PVC Pipe',
						'Concrete Pipe','Ordinary Wood Pipe','Cast Iron Pipe',
						'Galvanized Iron Pipe','Carbon Steel Pipe',
						'Aluminum Pipe','Copper Pipe')
abs_roughness = [0.625,0.005,0.0015,2,5,0.8,0.015,0.075,0.001,0.0015]
material = {materials:abs_roughness}

def get_float(prompt, lowest=None, highest=None):
	'''Reprompts input if the input is not an float or not in the range of values)'''
	while True:
		try:
			response = float((input(prompt)))
			if highest is None and response < lowest:
				print('Please enter a value greater than {}'.format(lowest))
			elif lowest is None and response > highest:
				print('Please enter a value less than {}'.format(highest))
			elif (highest is not None and response>highest) or (lowest is not None and response<lowest):
				print('Please choose between {} and {} only'.format(lowest,highest))
			else:
				return response
		except ValueError:
			print('Please enter a valid input')

def get_yesno(prompt):
	'''Reprompts input if the input is not a string or the string is not y or n'''
	while True:
		response = input(prompt)
		if response.lower()=='y':
			return True
		elif response.lower()=='n':
			return False
		else:
			print('Please enter y or n only')

def Colebrook(f, Dp, Vpipe, epsilon):
	'''Colebrooks-White Equation that computes Darcy-Wisebach friction factor'''
	Dp_mm = Dp*1000 #Colebrooks Formula, Epsilon/Dp(relative roughness) Epsilon is in mm
	Reynolds_Number = (rho*Dp*Vpipe)/mu
	return 1/(-2*(np.log10((2.51/(Reynolds_Number*np.sqrt(f)))+(epsilon/(3.7*Dp_mm)))))**2

def FPIM(x, tol, Dp, V, epsilon):
	'''Fixed Point Iteration Function that solves implicit function(Colebrook-White Equation)'''
	previous=0
	current=x
	while abs((current-previous)/current)>tol:
		previous=current
		current=Colebrook(current, Dp, V, epsilon)
		if current == 0:
			return 1
	return current

def Velocity(Hg, Dp, lp, f, n, Dn):
	''' Returns the Nozzle Velocity'''
	Dr = Dn/Dp #Diameter Ratio
	fcoeff = f*(lp/Dp) #Friction Coefficient
	return np.sqrt((2*g*Hg)/(1+(0.04*(n**2)*(Dr**4))+(fcoeff*(n**2)*(Dr**4))))

def Power(V, Q):
	''' Return the power received by the runner'''
	Hvelocity = (V**2)/(2*g) #Velocity Head
	return (gamma*Q*Hvelocity)/1000

def Efficiency(Velocity_optimal, Dn_optimal, omega):
	'''Returns Runner Efficiency'''
	k = 0.85 #Runner Coefficient
	beta2 = (165*np.pi/180) #Typical Beta 2 converted to radians
	Umean = Runner(Dn_optimal, omega) #Mean Blade Speed
	v = Umean/Velocity_optimal #Speed ratio
	Nr = (2*v)*(1-v)*(1-(k*np.cos(beta2))) #Runner Effiency Definition
	return Nr  #Runner Efficiency

def Runner(Dn_optimal,omega):
	''' returns blade speed of the pelton wheel'''
	omega = (np.pi/30)*omega #converts rpm to rad/s
	rwheel = (Dn_optimal*17.5)/2 #Typical range of Dwheel is 15x-20x Dnozzle
	return rwheel*omega

def Plotter(database, omega):
	'''Plots x vs y depending on the user's choice'''
	list_plot=['Power vs Diameter; Single Jet','Power vs Diameter; Multiple Jet',
				'Velocity vs Diameter; Single Jet','Velocity vs Diameter; Multiple Jet',
				'Flowrate vs Diameter; Single Jet','Flowrate vs Diameter; Multiple Jet',
				'Runner Efficiency vs Varying number of jets' ]
	if len(database)==8: #If assuming arbitrary friction factor
		x_nozzle_diameters = database[0]
		x_number_nozzles = database[1]
		y_velocity = database[2]
		y_flowrate = database[3]
		y_power = database[4]
		v_optimal = database[5]
		d_optimal = database[6]
		y_nrunner = database[7]
	else: #If user selects a material
		x_nozzle_diameters = database[0]
		x_number_nozzles = database[1]
		y_velocity = database[2]
		y_flowrate = database[3]
		y_power = database[4]
		y_friction = database[5]
		v_optimal = database[6]
		d_optimal = database[7]
		y_nrunner = database[8]
	while True:
		print('>>> PELTON WHEEL DESIGNER PLOTTER <<<')
		for i in range(len(list_plot)):
			print('{} - {}'.format(i,list_plot[i]))
		response = get_float('Which plot do you wish to see? ',lowest=0,highest=len(list_plot)-1)
		if response == 0:
			print('Max Power of {}kW at {}m Nozzle Diameter'.format(max(y_power[0]),x_nozzle_diameters[np.argmax(y_power[0])]))
			plt.plot(x_nozzle_diameters,y_power[0])
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Runner Power (kW)')
		elif response == 1:
			for x in range(len(y_power)):
				print('For {}Nozzles, Optimum Nozzle Diameter is {} mm'.format(x,(d_optimal[x]*1000)))
				plt.plot(x_nozzle_diameters,y_power[x],label='{} Nozzle/s'.format(x+1))
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Runner Power (kW)')
			plt.legend()
		elif response == 2:
			plt.plot(x_nozzle_diameters,y_velocity[0])
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Nozzle Velocity (m/s)')
		elif response == 3:
			for x in range(len(y_velocity)):
				plt.plot(x_nozzle_diameters,y_velocity[x],label='{} Nozzle/s'.format(x+1))
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Nozzle Velocity (m/s)')
			plt.legend()
		elif response == 4:
			plt.plot(x_nozzle_diameters,y_flowrate[0])
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Flowrate (m^3/s)')
		elif response == 5:
			for x in range(len(y_flowrate)):
				plt.plot(x_nozzle_diameters,y_flowrate[x],label='{} Nozzle/s'.format(x+1))
			plt.xlabel('Nozzle Diameter (m)')
			plt.ylabel('Flowrate (m^3/s)')
			plt.legend()
		elif response == 6:
			for x in range(len(y_nrunner)):
				print('For {} rpm, Optimum number of jets is {}'.format(omega[x], x_number_nozzles[np.argmax(y_nrunner[x])]))
				plt.plot(x_number_nozzles,y_nrunner[x],label='{} rpm'.format(omega[x]))
			plt.xlabel('Number of nozzles')
			plt.ylabel('Runner Efficiency')
			plt.legend()
			plt.ylim(0,1)
		plt.show()
		cont = get_yesno('Continue Plotting? [y/n] ')
		if cont == True:
			pass
		else:
			break


def Designer(Hr, Hn, Dp, lp, f, N, omega, *epsilon):
	'''Return arrays of Nozzle Diameter, Velocity, Power and Hydraulic Efficiency'''
	Hg = Hr-Hn #Gross Head
	Dn = np.linspace(0.00001, Dp, num=N, endpoint=True) #Evenly spaced array of possible Nozzle Diameter
	nozzles = np.arange(1,11)
	velocities = []
	flowrate = []
	power = []
	F=[]
	nrunner = []
	if not epsilon:
		for n in nozzles:
			V = Velocity(Hg, Dp, lp, f, n, Dn) #Array of Velocity for n nozzles
			Q = n*(np.pi)*(Dn**2)*V*0.25 #Array of Flowrate for n nozzles
			Pideal= Power(V, Q) #Array of Runner Power for n nozzles
			velocities.append(V)
			flowrate.append(Q)
			power.append(Pideal)
		'''arrays of velocity and nozzle diameter at max power per number of nozzle'''
		v_optimal=np.array([velocities[x][np.argmax(power[x])] for x in range(len(power))])
		d_optimal=np.array([Dn[np.argmax(power[x])] for x in range(len(power))])
		'''Computes for runner efficiency'''
		for rpms in omega:
			NR=[]
			for index in range(len(d_optimal)):
				Nrunner= Efficiency(v_optimal[index], d_optimal[index], rpms)
				NR.append(Nrunner)
			nrunner.append(NR)
		return Dn, nozzles, velocities, flowrate, power, v_optimal, d_optimal, nrunner
	else:
		epsilon=epsilon[0]
		for n in nozzles:
			V=[]
			friction=[]
			for diameter in Dn:
				Dr = diameter/Dp
				v = Velocity(Hg, Dp, lp, f, n, diameter) #Array of Velocity for n nozzles
				v_pipe = v*(Dr**2)*n
				f = FPIM(f, 0.000001, Dp, v_pipe, epsilon) #solve for the new f
				V.append(v)
				friction.append(f)
			Q = n*(np.pi)*(Dn**2)*np.array(V)*0.25 #Array of Flowrate for n nozzles
			Pideal= Power(np.array(V), Q) #Array of Runner Power for n nozzles
			velocities.append(np.array(V))
			flowrate.append(np.array(Q))
			power.append(np.array(Pideal))
			F.append(np.array(friction))
		'''arrays of velocity and nozzle diameter at max power per number of nozzle'''
		v_optimal=[velocities[x][np.argmax(power[x])] for x in range(len(power))]
		d_optimal=[Dn[np.argmax(power[x])] for x in range(len(power))]
		'''Computes for runner efficiency'''
		for rpms in omega:
			NR=[]
			for index in range(len(d_optimal)):
				Nrunner= Efficiency(v_optimal[index], d_optimal[index], rpms)
				NR.append(Nrunner)
			nrunner.append(NR)
		return  Dn, nozzles, velocities, flowrate, power, F, v_optimal, d_optimal, nrunner

def Initialize(database):
	'''Initialize with user input'''
	print('>>> INITIALIZING PELTON WHEEL DESIGNER <<<')
	Hr = get_float('Input Reservoir Elevation: ',lowest=0) #Define Resevoir Elevation
	Hn = get_float('Input Nozzle Elevation: ',lowest=0) #Define Nozzle Elevation
	Dp = get_float('Input Penstock Diameter: ',lowest=0) #Define Penstock Diameter
	lp = get_float('Input Penstock Length: ',lowest=0) #Define Penstock Length
	N = get_float('Define mesh size: ',lowest=1) #Define Iteration mesh size
	omega_lower = get_float('Define lower operating wheel rpm: ',lowest=0) #Lower bound of wheel rpm
	omega_upper = get_float('Define upper operating wheel rpm: ',lowest=1) #Upper bound of wheel rpm
	omega_increment = get_float('Define wheel rpm increment: ',lowest=1, highest=(omega_upper-1)) #Wheel increment
	omega = np.arange(omega_lower,omega_upper,omega_increment) #Array of Wheel rpms

	response = get_yesno('Assume arbitrary friction factor(f)? [y/n] ') #Y or N
	if response == True:
		f = get_float('Input friction factor: ', lowest=0) #Define Darcy-Wisebach friction factor
		print('\n','<<< INITIALIZATION COMPLETE >>>')
		print('COMPUTING...')
		Dn,nozzles, velocities, flowrate, Power, v_optimal, d_optimal, nrunner=Designer(Hr, Hn, Dp, lp, f, N, omega)
		database.append(Dn)
		database.append(nozzles)
		database.append(velocities)
		database.append(flowrate)
		database.append(Power)
		database.append(v_optimal)
		database.append(d_optimal)
		database.append(nrunner)
	else:
		'''Compute for a more realistic friction factor according to the material'''
		print("Please select your piping material:")
		print('Source: https://www.enggcyclopedia.com/2011/09/absolute-roughness/','\n')
		for i in range(len(materials)):
			print('{} - {} [absolute roughness: {} mm]'.format(i,materials[i],abs_roughness[i]))
		choice = get_float("Material no: ",lowest=0, highest=9)
		epsilon = abs_roughness[int(choice)]
		print('\n','You have chosen {} with absolute roughness of {}mm'.format(materials[int(choice)],abs_roughness[int(choice)]),'\n')
		f = get_float("What is the guess value of the friction factor? ", lowest=0)
		print('\n','<<< INITIALIZATION COMPLETE >>>')
		print('COMPUTING...')
		Dn, nozzles, velocities, flowrate, Power, F, v_optimal, d_optimal, nrunner=Designer(Hr, Hn, Dp, lp, f, N, omega, epsilon)
		database.append(Dn)
		database.append(nozzles)
		database.append(velocities)
		database.append(flowrate)
		database.append(Power)
		database.append(F)
		database.append(v_optimal)
		database.append(d_optimal)
		database.append(nrunner)
	Plotter(database, omega)
	return database

database=[]
db = Initialize(database)
