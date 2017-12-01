#pragma once

#ifndef calculation
#define calculation



#define number_hidden_neurons 5
#define number_hidden_layers 10
#define number_imports 2
#define number_outputs 1
#define iterations 10000


double sigmoid(double x) { return 1 / (1 + exp(-x)); }

struct neuron
{
	long double *hidden_dc_dz;
	long double *hidden_outputs;
	long double *hidden_weights;
	void creat_neuron();
};

struct layer
{
	void creat_layer();
	neuron **hidden_neurons;
};

void neuron::creat_neuron() {

	float sign = -1;
	float random;
	hidden_weights = new long double[number_hidden_neurons];
	hidden_outputs = new long double[number_hidden_neurons];
	hidden_dc_dz = new long double[number_hidden_neurons];

	for (int i = 0; i<number_hidden_neurons; i++)
	{
		random = (float(rand()) / float(RAND_MAX)) / 2.f;
		random *= sign;
		sign *= -1;
		hidden_weights[i] = random;
		hidden_outputs[i] = 0;
		hidden_dc_dz[i] = 0;
	}
}

void layer::creat_layer()
{
	int i;
	hidden_neurons = new neuron*[number_hidden_neurons];

	for (i = 0; i<number_hidden_neurons; i++)
	{
		hidden_neurons[i] = new neuron;

		hidden_neurons[i]->creat_neuron();

	}
}
// for last layer

struct output_neuron
{
	long double *output_dc_dz;
	long double *output_outputs;
	long double *output_weights;
	void creat_output_neuron();
};

struct output_layer
{
	void creat_output_layer();
	output_neuron **output_neurons;
};

void output_neuron::creat_output_neuron() {

	float sign = -1;
	float random;

	//output neuorn的weight有10個  output有兩個
	output_weights = new long double[number_hidden_neurons];
	output_outputs = new long double[number_hidden_neurons];
	output_dc_dz = new long double[number_hidden_neurons];
	for (int i = 0; i<number_hidden_neurons; i++)
	{
		random = (float(rand()) / float(RAND_MAX)) / 2.f;
		random *= sign;
		sign *= -1;
		output_weights[i] = random;
		output_outputs[i] = 0;
		output_dc_dz[i] = 0;
	}
}

void output_layer::creat_output_layer()
{
	int i;
	output_neurons = new output_neuron*[number_outputs];
	for (i = 0; i<1; i++)
	{
		output_neurons[i] = new output_neuron;
		output_neurons[i]->creat_output_neuron();
	}
}

class calculate
{
	double lamda = 0.005;
	layer hidden_layer_object[number_hidden_layers];
	output_layer output_layer_object[1];   //output layer just one
	neuron *layer[1];

public:
	void initialize(); 	// creat hidden layer
	void forward(std::vector<double> input); //input is input data,using vector container
	void backword(std::vector<double> input, double target_output);//target_output is the data you want
};

void calculate::initialize() {




	// creat hidden layer
	for (size_t i = 0; i < number_hidden_layers; i++)
	{
		hidden_layer_object[i].creat_layer();
	}
	output_layer_object[0].creat_output_layer();
}

void calculate::forward(vector<double> input) {

	//output in first hidden layer
	for (size_t i = 0; i < 1; i++)
	{
		for (size_t j = 0; j < number_hidden_neurons; j++)  //第幾個neuron
		{
			double num = 0;

			for (size_t k = 0; k < input.size(); k++) // input * weights of j-th neuron
			{
				num += input[k] * hidden_layer_object[i].hidden_neurons[j]->hidden_weights[k];
			}
			hidden_layer_object[i].hidden_neurons[j]->hidden_outputs[j] = sigmoid(num);
		}
	}

	//  output inside hidden layer 

	for (size_t i = 1; i < number_hidden_layers; i++)
	{
		for (size_t j = 0; j < number_hidden_neurons; j++)
		{
			double num = 0;
			for (size_t k = 0; k < number_hidden_neurons; k++)
			{
				num += hidden_layer_object[i - 1].hidden_neurons[k]->hidden_outputs[k] *
					hidden_layer_object[i].hidden_neurons[j]->hidden_weights[k];
			}
			hidden_layer_object[i].hidden_neurons[j]->hidden_outputs[j] = sigmoid(num);
		}
	}


	// output layer

	double nnum = 0;
	for (size_t i = 0; i < number_hidden_neurons; i++)
	{
		nnum += output_layer_object[0].output_neurons[0]->output_weights[i] *
			hidden_layer_object[4].hidden_neurons[i]->hidden_outputs[i];
	}
	output_layer_object[0].output_neurons[0]->output_outputs[0] = sigmoid(nnum);
}

void calculate::backword(vector<double> input, double target_output) {

	double actual_output = output_layer_object[0].output_neurons[0]->output_outputs[0];

	//output layer
	double n;

	// dy_dz
	double dy_dz_pron = output_layer_object[0].output_neurons[0]->output_outputs[0] *
		(1 - (output_layer_object[0].output_neurons[0]->output_outputs[0]));

	for (size_t i = 0; i < number_hidden_neurons; i++)
	{
		//chain rule
		n = dy_dz_pron*hidden_layer_object[number_hidden_layers - 1].hidden_neurons[i]->hidden_outputs[i];
		//dc_dz
		hidden_layer_object[number_hidden_layers - 1].hidden_neurons[i]->hidden_dc_dz[i] = n;
		//dc_dw
		double dc_dw = n*(actual_output - target_output);
		//updata_weight
		output_layer_object[0].output_neurons[0]->output_weights[i] -= lamda*dc_dw;
	}


	// update last hidden layer weight

	double delta1; // i_j
	double dc_dwij;
	for (size_t j = 0; j < number_hidden_neurons; j++)
	{
		delta1 = (actual_output - target_output)*(output_layer_object[0].output_neurons[0]->output_outputs[0])*
			(1 - (output_layer_object[0].output_neurons[0]->output_outputs[0]))*
			output_layer_object[0].output_neurons[0]->output_weights[j] *
			(hidden_layer_object[4].hidden_neurons[j]->hidden_outputs[j])*
			(1 - (hidden_layer_object[4].hidden_neurons[j]->hidden_outputs[j]));

		for (size_t i = 0; i < number_hidden_neurons; i++)
		{
			dc_dwij = delta1*hidden_layer_object[3].hidden_neurons[i]->hidden_outputs[i];
			hidden_layer_object[4].hidden_neurons[j]->hidden_weights[i] -= lamda*dc_dwij;
		}
	}

	// hidden layer
	for (size_t k = number_hidden_layers - 2; k <= 0; k--)
	{
		for (size_t i = 0; i < number_hidden_neurons; i++)
		{
			double nnn = 0;

			for (size_t j = 0; j < number_hidden_neurons; j++)
			{
				nnn += hidden_layer_object[k + 1].hidden_neurons[j]->hidden_dc_dz[j] *
					hidden_layer_object[k + 1].hidden_neurons[j]->hidden_weights[i];
			}
			hidden_layer_object[k].hidden_neurons[i]->hidden_dc_dz[i] = nnn*
				(hidden_layer_object[k].hidden_neurons[i]->hidden_outputs[i])*
				(1 - (hidden_layer_object[k].hidden_neurons[i]->hidden_outputs[i]));
		}
	}
	// dc_dw

	double f_pron, m;

	for (size_t i = number_hidden_layers - 2; i <= 1; i--)  // layer
	{
		for (size_t j = 0; j < number_hidden_neurons; j++)  //neuron
		{
			f_pron = hidden_layer_object[i].hidden_neurons[j]->hidden_dc_dz[j];

			for (size_t k = 0; k < number_hidden_neurons; k++) // weight per neuron
			{
				m = f_pron*hidden_layer_object[i - 1].hidden_neurons[k]->hidden_outputs[k];
				hidden_layer_object[i].hidden_neurons[j]->hidden_weights[k] -= lamda*m*f_pron;
			}
		}
	}

	// first hidden layer

	double mm, dc_dew;

	for (size_t i = 0; i < number_hidden_neurons; i++)
	{
		mm = hidden_layer_object[0].hidden_neurons[i]->hidden_dc_dz[i];

		for (size_t j = 0; j < input.size(); j++)
		{
			dc_dew = mm*input[j];
			hidden_layer_object[0].hidden_neurons[i]->hidden_weights[j] -= lamda*dc_dew;
		}
	}



	for (size_t i = 0; i < number_hidden_neurons; i++)
	{
		cout << output_layer_object[0].output_neurons[0]->output_weights[i] << "   ";
	}
	cout << endl;
};

#endif






