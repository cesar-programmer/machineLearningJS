/* eslint-disable no-undef */
/* eslint-disable no-tabs */
let stopTraining = false
// aqui me traigo los datos de la api
async function getData () {
  const datosCasasR = await fetch('https://static.platzi.com/media/public/uploads/datos-entrenamiento_15cd99ce-3561-494e-8f56-9492d4e86438.json')
  const datosCasas = await datosCasasR.json()
  // limpiar los datos que no tienen precio o cuartos de la casa
  // crear un nuevo objeto con los datos que si tienen precio y cuartos y aplico un filtro
  const datosLimpios = datosCasas.map(casa => ({
    precio: casa.Precio,
    cuartos: casa.NumeroDeCuartosPromedio
  }))
  // si el precio o los cuartos son null no los voy a tener en cuenta
    .filter(casa => (casa.precio != null && casa.cuartos != null))

  return datosLimpios
}

// aqui visualizo los datos en una grafica de puntos con tfvis
function visualizarDatos (data) {
  const valores = data.map(d => ({
    x: d.cuartos,
    y: d.precio
  }))

  tfvis.render.scatterplot(
    { name: 'Cuartos vs Precio' },
    { values: valores },
    {
      xLabel: 'Cuartos',
      yLabel: 'Precio',
      height: 300
    }
  )
}

// aqui creo el modelo para entrenarlo con los datos de la api y lo retorno
function crearModelo () {
  // sequencial es una funcion que me permite crear un modelo de red neuronal secuencial
  // que es una red neuronal que tiene una capa de entrada, una capa oculta y una capa de salida
  const modelo = tf.sequential()

  // esta es la capa de entrada que va a tener 1 sola unidad la cual va a recibir un solo valor
  // el valor a recibir es el numero de cuartos de la casa y el valor a predecir es el precio de la casa
  modelo.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))

  // esta es la capa oculta que va a tener 1 sola unidad y va a recibir un solo
  // valor que es el valor de la capa de entrada y va a predecir el valor de la capa de salida
  modelo.add(tf.layers.dense({ units: 1, useBias: true }))

  return modelo
}

// entrenar el modelo con los datos de la api
async function entrenarModelo (modelo, entradas, salidas) {
  // aqui le digo al modelo que va a recibir un valor de entrada y va a predecir un valor de salida
  modelo.compile({
    // optimizador que me permite optimizar el modelo
    optimizer: tf.train.adam(),
    // funcion de perdida que me permite saber que tan bien o mal esta aprendiendo el modelo
    loss: tf.losses.meanSquaredError,
    // metrica que me permite saber que tan bien o mal esta aprendiendo el modelo
    metrics: ['mse']
  })

  // aqui le digo al modelo que entrene con los datos de entrada y salida
  // y que lo haga 50 veces
  const surface = { name: 'show.history live', tab: 'Training' }
  const tamanoBatch = 28
  const epochs = 50
  const history = []

  return await modelo.fit(entradas, salidas, {
    // tamaño de los datos de entrada
    batchSize: tamanoBatch,
    // numero de veces que se va a entrenar el modelo
    epochs,
    // callbacks que me permite visualizar los datos de entrenamiento
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        history.push(logs)
        tfvis.show.history(surface, history, ['loss', 'mse'])
        if (stopTraining) {
          modelo.stopTraining = true
        }
      }
    }
  })
}

// almacenar el modelo en el disco duro
async function guardarModelo () {
  const saveResult = await modelo.save('downloads://modelo-regresion-lineal')
  console.log(saveResult)
}

// cargar el modelo del disco duro
async function cargarModelo () {
  const uploadJSONInput = document.getElementById('upload-json')
  const uploadWeightsInput = document.getElementById('upload-weights')

  modelo = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]))
  console.log('modelo cargado')
}

// mostrar curva de inferencia (prediccion)
async function mostrarCurvaInferencia () {
  const data = await getData()
  const tensorData = await convertirDatosATensores(data)

  const { entradasMax, entradasMin, salidasMax, salidasMin } = tensorData

  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = modelo.predict(xs.reshape([100, 1]))

    const desnormX = xs
      .mul(entradasMax.sub(entradasMin))
      .add(entradasMin)

    const desnormY = preds
      .mul(salidasMax.sub(salidasMin))
      .add(salidasMin)

    return [desnormX.dataSync(), desnormY.dataSync()]
  })

  const puntosPrediccion = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const puntosOriginales = data.map(d => ({
    x: d.cuartos, y: d.precio
  }))

  tfvis.render.scatterplot(
    { name: 'Prediccion vs Originales' },
    { values: [puntosOriginales, puntosPrediccion], series: ['originales', 'predicciones'] },
    {
      xLabel: 'Cuartos',
      yLabel: 'Precio',
      height: 300
    }
  )
}

function convertirDatosATensores (data) {
  // tf.tidy es una funcion que me permite limpiar la memoria de los tensores
  return tf.tidy(() => {
    // desordenar los datos para que no se aprenda en un orden especifico
    tf.util.shuffle(data)
    // convertir los datos a tensores
    // los tensores son un tipo de dato que se usa en tensorflow para poder entrenar el modelo
    const entradas = data.map(d => d.cuartos)
    const tensorEntradas = tf.tensor2d(entradas, [entradas.length, 1])

    // convertir los datos de salida a un tensor de 1 dimension
    const salidas = data.map(d => d.precio)
    const tensorSalidas = tf.tensor2d(salidas, [salidas.length, 1])

    const entradasMax = tensorEntradas.max()
    const entradasMin = tensorEntradas.min()
    const salidasMax = tensorSalidas.max()
    const salidasMin = tensorSalidas.min()

    // normalizar los datos de entrada y salida
    // normalizar es convertir los datos a un rango de 0 a 1
    const entradasNormalizadas = tensorEntradas.sub(entradasMin).div(entradasMax.sub(entradasMin))
    const salidasNormalizadas = tensorSalidas.sub(salidasMin).div(salidasMax.sub(salidasMin))

    return {
      entradas: entradasNormalizadas,
      salidas: salidasNormalizadas,
      entradasMax,
      entradasMin,
      salidasMax,
      salidasMin
    }
  })
}

// aqui entreno el modelo con los datos de la api
let modelo
async function run () {
  const data = await getData()

  visualizarDatos(data)

  modelo = crearModelo()
  const tensorData = convertirDatosATensores(data)
  const { entradas, salidas } = tensorData
  entrenarModelo(modelo, entradas, salidas)
}

run()
