import random #Importa la librería random de Python, que se utiliza para generar números aleatorios
import numpy as np #Se agregó la librería numpy y la renombramos con np

class CrossEntropyCost(object): #Definimos la Clase CroosEntropy
    def fn(a, y): #calcula el costo de la entropía entre las salidas predichas a y las salidas reales y.
        #La entropía cruzada es una medida de la diferencia entre dos distribuciones de probabilidad, en este
        #caso, la distribución de probabilidad predicha a y la distribución de probabilidad real y.
        #np.nan_to_num se utiliza para evitar problemas con el logaritmo de 0.
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def delta(z, a, y): #calcula el error de la capa de salida en una red neuronal. Este error se utiliza
        #durante el algoritmo de retropropagación para actualizar los pesos y sesgos de la red.
        return (a-y)

class Network(object): #Se definió una clase

    def __init__(self, sizes, cost=CrossEntropyCost): #Es el constructor de la clase Network. 
        #Toma un argumento sizes que es una lista de enteros que representa
        #el número de neuronas en cada capa de la red neuronal
        self.cost=cost # Es la función de costo que se utilizará, por defecto es CrossEntropyCost
        self.num_layers = len(sizes) #Número de capas en la red neurona
        self.sizes = sizes #Es una lista que contiene el número de neuronas en cada capa
       self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #Es una lista de matrices
        #de b´s para cada capa de la red neuronal
        self.weights = [np.random.randn(y, x) #Es una lista de matrices de w´s para cada capa de la red neuronal
                        for x, y in zip(sizes[:-1], sizes[1:])] 

    def feedforward(self, a): #Toma un argumento "a" que es una matriz de entrada para la red
        #neuronal, realiza todos los procesos dentro de las capas ocultas hacia adelante a través de la red neuronal
        #y devuelve la salida de la red neuronal.
        for b, w in zip(self.biases, self.weights): #Une ambas matrices en una
            a = sigmoid(np.dot(w, a)+b) #Aplica la función sigmoide a la multiplicación de a por w sumada con b 
        return a #Regresa "a"

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Es un algoritmo que utiliza una muestra aleatoria de datos de entrenamiento
                #en cada iteración para actualizar los w´s y b´s de la red neuronal hasta alcanzar un minimó
                #local en la función de perdida.
        if test_data:
            test_data = list(test_data) #Datos de prueba, lista de duplas de entrada y salida utilizadas para probar 
            #la red neuronal (b´s y w´s).
            n_test = len(test_data) #Nos da el numero de elementos en la lista

        training_data = list(training_data) #Datos de entrenamiento, lista de duplas de entrada y salida utilizadas 
                #para probar la red neuronal (b´s y w´s).
        n = len(training_data) #Nos da el numero deelementos en la lista
        for j in range(epochs): #Ciclo for que se ejecuta epochs veces
            random.shuffle(training_data) #Se mezcla aleatoriamente el conjunto de datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #Luego, se divide el conjunto de datos de entrenamiento
            for mini_batch in mini_batches: #Itera sobre cada mini-batche y llama al método update_mini_batch
                #para actualizar los w´s y b´s de la red neuronal utilizando el algoritmo SGD.
                self.update_mini_batch(mini_batch, eta) 
            if test_data: #Si test_data no es 0, se imprime el número de aciertos en los datos de prueba para la
                #época actual. De lo contrario, se imprime un mensaje indicando que se ha completado la época
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta): #Definamos la función que se uso anteriormente para actualizar
        #los w´s y b´s 
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Llena la matriz nabla_b de ceros con np.zeros
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Llena la matriz nabla_w de ceros con np.zeros
        for x, y in mini_batch: #Itera sobre cada elemento del mini-batch y utiliza
            #backpropagation para calcular los gradientes de los w´s y b´s.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Se suman a la matriz nabla b
            #los gradientes de cada una de las b´s
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #Se suman a la matriz nabla w
            #los gradientes de cada una de las w´s
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] #Los w´s se actualizan restandole a los pesos
            #la multiplicación de nabla_w por la división de la taza de aprendizaja con el tamaño del mini_batch
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] #Los b´s se actualizan restandole a los biases
            #la multiplicación de nabla_b por la división de la taza de aprendizaja con el tamaño del mini_batch

    def backprop(self, x, y): #Definimos backprop, es utilizada para calcular el gradiente de la función de costo
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Llena la matriz nabla_b de ceros con np.zeros
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Llena la matriz nabla_w de ceros con np.zeros
        # feedforward
        activation = x 
        activations = [x] #Lista para almacenar todas las activaciones, capa por capa.
        zs = [] #Lista para almacenar todos los vectores z, capa por capa.
        for b, w in zip(self.biases, self.weights): #Por cada b, w en la matriz formada al unir las matrices
            #biases y weigths se definira z
            z = np.dot(w, activation)+b #z es la suma de la multiplicación de cada w  con su respectiva activación
            #sumada b
            zs.append(z) #Se almacena en la lista zs
            activation = sigmoid(z) #Función de activación
            activations.append(activation) #Se almacenan en la lista de activations
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y) * sigmoid_prime(zs[-1]) #Llama a la función delta de la
        #clase de costo para calcular el error en la salida de la red neuronal, calculandose como la multiplicación de la
        #diferencia entre la salida de la red neuronal y la salida esperada por la derivada de la función de activación
        #sigmoid evaluada en la última capa de la red neuronal.     
        nabla_b[-1] = delta #Define que es nabla_b en la última capa 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Define que es nabla_w en la última capa
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #El ciclo itera sobre cada una de las capas de la red
            z = zs[-l] #Definimos z como la z de la última capa
            sp = sigmoid_prime(z) #llamamos sp a la derivada de la sigmoide evaluada en z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Recalculamos delta como la multiplicación de 
            #la matriz traspuesta de w´s de la capa que sigue por el delta de la capa anterior por sp
            nabla_b[-l] = delta # Redefinimos la ultima capa de nabla_b como lo calculado
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Redefinimos nabla_w de la ultima capa
        return (nabla_b, nabla_w) #Regresa nabla_b y nabla_w una vez haya pasado de la ultima capa a la primer capa

    def evaluate(self, test_data): #La función evaluate es utilizada para evaluar el rendimiento de la red neuronal
        #utilizando un conjunto de datos de prueba.
        test_results = [(np.argmax(self.feedforward(x)), y) #Es una lista de duplas que contiene la predicción de la red
                        #neuronal y la salida esperada para cada elemento del conjunto de datos de prueba.
                        for (x, y) in test_data] 
        return sum(int(x == y) for (x, y) in test_results) #Nos da el numero de aciertos que tuvimos 

    def cost_derivative(self, output_activations, y): #Mide la diferencia entre la salida de la red neuronal y la
        #salida esperada.
        return (output_activations-y) 

#### Miscellaneous functions
def sigmoid(z): #Función sigmoide/de activación
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): #Derivada calculada a mano de la función sigmoide
    return sigmoid(z)*(1-sigmoid(z))
