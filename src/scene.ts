import GUI from 'lil-gui'
import * as THREE from 'three/webgpu'
import {
  Fn,
  If,
  Return,
  instancedArray,
  instanceIndex,
  uniform,
  select,
  attribute,
  uint,
  Loop,
  float,
  transformNormalToView,
  cross,
  triNoise3D,
  time,
} from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { DragControls } from 'three/addons/controls/DragControls.js'
import WebGPU from 'three/addons/capabilities/WebGPU.js'
import Stats from 'stats.js'
import { toggleFullScreen } from './helpers/fullscreen'
import { resizeRendererToDisplaySize } from './helpers/responsiveness'
import './style.css'

const CANVAS_ID = 'scene'

let canvas: HTMLCanvasElement
let renderer: THREE.WebGPURenderer
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let cameraControls: OrbitControls
let dragControls: DragControls
let clock: THREE.Clock
let stats: Stats
let gui: GUI

// Lights and Helpers
let ambientLight: THREE.AmbientLight
let pointLight: THREE.PointLight
let axesHelper: THREE.AxesHelper
let pointLightHelper: THREE.PointLightHelper
let gridHelper: THREE.GridHelper

// Cloth simulation parameters
let clothWidth = 1
let clothHeight = 1
const clothNumSegmentsX = 12
const clothNumSegmentsY = 12
const baseSize = 1 // Base dimension for fitting
// Box dimensions (width x height x depth)
const boxWidth = 0.8
const boxHeight = 0.8
const boxDepth = 0.05

let vertexPositionBuffer: any,
  vertexForceBuffer: any,
  vertexParamsBuffer: any
let springVertexIdBuffer: any,
  springRestLengthBuffer: any,
  springForceBuffer: any
let springListBuffer: any
let computeSpringForces: any, computeVertexForces: any
let dampeningUniform: any,
  boxPositionUniform: any,
  stiffnessUniform: any,
  boxUniform: any,
  windUniform: any,
  boxSizeUniform: any,
  anchorPositionUniform: any,
  gravityUniform: any
let vertexWireframeObject: THREE.Mesh,
  springWireframeObject: THREE.Line
let clothMesh: THREE.Mesh, clothMaterial: any, box: THREE.Mesh, anchorBox: THREE.Mesh
let clothTexture: THREE.Texture | null = null
let textureAspectRatio = 1
let timeSinceLastStep = 0
let timestamp = 0
const verletVertices: any[] = []
const verletSprings: any[] = []
const verletVertexColumns: any[] = []

// Available poster images
// Using import.meta.env.BASE_URL to handle both dev and production paths
const availablePosters = {
  'Poster 1': `${import.meta.env.BASE_URL}posters/gare-poster-1.jpg`,
  'Poster 2': `${import.meta.env.BASE_URL}posters/gare-poster-2.jpg`,
  'Poster 3': `${import.meta.env.BASE_URL}posters/gare-poster-3.jpg`,
  'circuito': `${import.meta.env.BASE_URL}posters/circuito.png`,
}

const params = {
  wireframe: false,
  boxEnabled: true, // Controls physics collision
  boxVisible: true, // Controls mesh visibility
  wind: 1.0,
  anchorZ: 0, // Z position of the anchor box
  dampening: 0.95, // Energy loss per frame (lower = more damped/heavy feeling)
  gravity: 0.0002, // Gravity strength (higher = heavier cloth)
  useTexture: true, // Use image texture instead of solid color
  fitMode: 'width', // 'width' or 'height' - which dimension to fit
  useColorTint: false, // Apply color tinting to texture
  opacity: 1.0, // Cloth opacity (0 = transparent, 1 = opaque)
  selectedPoster: 'Poster 1', // Currently selected poster
}

const API = {
  color: 0x204080, // sRGB
  sheenColor: 0xffffff, // sRGB
}

// Check for WebGPU support
if (WebGPU.isAvailable() === false) {
  document.body.appendChild(WebGPU.getErrorMessage())
  throw new Error('No WebGPU support')
}

/**
 * Calculate cloth dimensions based on texture aspect ratio and fit mode
 */
function calculateClothDimensions() {
  if (params.fitMode === 'width') {
    // Fit to width, adjust height based on aspect ratio
    clothWidth = baseSize
    clothHeight = baseSize / textureAspectRatio
  } else {
    // Fit to height, adjust width based on aspect ratio
    clothHeight = baseSize
    clothWidth = baseSize * textureAspectRatio
  }
}


init()

async function init() {
  // ===== 📋 LOAD SAVED SETTINGS =====
  {
    // Load saved poster selection from localStorage
    const savedPoster = localStorage.getItem('selectedPoster')
    if (savedPoster && availablePosters[savedPoster as keyof typeof availablePosters]) {
      params.selectedPoster = savedPoster
      console.log(`Loading saved poster: ${savedPoster}`)
    }
  }

  // ===== 🖼️ CANVAS, RENDERER, & SCENE =====
  {
    canvas = document.querySelector(`canvas#${CANVAS_ID}`)!
    renderer = new THREE.WebGPURenderer({ canvas, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.toneMapping = THREE.NeutralToneMapping
    renderer.toneMappingExposure = 1
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
  }

  // ===== 🖼️ LOAD TEXTURE =====
  {
    const textureLoader = new THREE.TextureLoader()
    const posterPath = availablePosters[params.selectedPoster as keyof typeof availablePosters]
    clothTexture = await textureLoader.loadAsync(posterPath)
    clothTexture.colorSpace = THREE.SRGBColorSpace
    
    // Calculate aspect ratio from loaded image
    if (clothTexture.image) {
      textureAspectRatio = clothTexture.image.width / clothTexture.image.height
      calculateClothDimensions()
      console.log(`Texture loaded: ${params.selectedPoster} (${clothTexture.image.width}x${clothTexture.image.height})`)
      console.log(`Aspect ratio: ${textureAspectRatio.toFixed(2)}, Cloth dimensions: ${clothWidth.toFixed(2)}x${clothHeight.toFixed(2)}`)
    }
  }

  // ===== 💡 LIGHTS =====
  {
    ambientLight = new THREE.AmbientLight('white', 0.4)
    scene.add(ambientLight)

    pointLight = new THREE.PointLight('white', 20, 100)
    pointLight.position.set(-2, 2, -2)
    pointLight.castShadow = true
    pointLight.shadow.radius = 4
    pointLight.shadow.camera.near = 0.1
    pointLight.shadow.camera.far = 1000
    pointLight.shadow.mapSize.width = 2048
    pointLight.shadow.mapSize.height = 2048
    scene.add(pointLight)
  }

  // ===== 🎥 CAMERA =====
  {
    camera = new THREE.PerspectiveCamera(
      40,
      window.innerWidth / window.innerHeight,
      0.01,
      10
    )
    camera.position.set(-1.6, -0.1, -1.6)
  }

  // ===== 🕹️ CONTROLS =====
  {
    cameraControls = new OrbitControls(camera, renderer.domElement)
    cameraControls.minDistance = 1
    cameraControls.maxDistance = 8
    cameraControls.target.set(0, -0.1, 0)
    cameraControls.enableDamping = true
    cameraControls.autoRotate = false
    cameraControls.update()

    // Full screen
    window.addEventListener('dblclick', (event) => {
      if (event.target === canvas) {
        toggleFullScreen(canvas)
      }
    })

    window.addEventListener('resize', onWindowResize)
  }

  // ===== 🪄 HELPERS =====
  {
    axesHelper = new THREE.AxesHelper(4)
    axesHelper.visible = false
    scene.add(axesHelper)

    pointLightHelper = new THREE.PointLightHelper(
      pointLight,
      undefined,
      'orange'
    )
    pointLightHelper.visible = false
    scene.add(pointLightHelper)

    gridHelper = new THREE.GridHelper(20, 100, 'teal', 'darkgray')
    gridHelper.position.y = -0.5
    scene.add(gridHelper)
  }

  // ===== 📈 STATS & CLOCK =====
  {
    clock = new THREE.Clock()
    stats = new Stats()
    document.body.appendChild(stats.dom)
  }

  // ===== 📦 SETUP CLOTH SIMULATION =====
  setupCloth()

  // ===== 🕹️ DRAG CONTROLS (after box is created) =====
  {
    dragControls = new DragControls([box], camera, renderer.domElement)
    dragControls.addEventListener('hoveron', () => {
      cameraControls.enabled = false
    })
    dragControls.addEventListener('hoveroff', () => {
      cameraControls.enabled = true
    })
    dragControls.addEventListener('dragstart', () => {
      cameraControls.enabled = false
    })
    dragControls.addEventListener('dragend', () => {
      cameraControls.enabled = true
    })
    dragControls.enabled = false
  }

  // ==== 🐞 DEBUG GUI ====
  {
    gui = new GUI({ title: '🐞 Cloth Simulation', width: 300 })

    // Cloth Physics Folder
    const physicsFolder = gui.addFolder('Cloth Physics')
    physicsFolder.add(stiffnessUniform, 'value', 0.1, 1.0, 0.01).name('stiffness')
    physicsFolder.add(params, 'dampening', 0.9, 0.99, 0.001).name('dampening').onChange(() => {
      // Lower dampening = more energy loss = heavier, slower cloth
    })
    physicsFolder.add(params, 'gravity', 0.00001, 0.001, 0.00001).name('gravity').onChange(() => {
      // Higher gravity = heavier cloth
    })
    physicsFolder.add(params, 'wind', 0, 5, 0.1).name('wind force')
    physicsFolder.add(params, 'anchorZ', -2, 2, 0.01).name('anchor Z position')
    physicsFolder.add(params, 'wireframe').name('show wireframe')

    // Cloth Material Folder
    const materialFolder = gui.addFolder('Cloth Material')
    
    materialFolder.add(params, 'selectedPoster', Object.keys(availablePosters))
      .name('poster image')
      .onChange(async (posterName: string) => {
        // Save selection to localStorage
        localStorage.setItem('selectedPoster', posterName)
        // Reload page to apply new aspect ratio
        console.log(`Switching to ${posterName}... Reloading to apply aspect ratio.`)
        window.location.reload()
      })
    
    materialFolder.add(params, 'useTexture').name('use texture').onChange((value: boolean) => {
      clothMaterial.map = value ? clothTexture : null
      // Update color based on texture and tint settings
      const newColor = (value && !params.useColorTint) ? 0xffffff : API.color
      clothMaterial.color.setHex(newColor)
      clothMaterial.needsUpdate = true
    })
    
    materialFolder.add(params, 'useColorTint').name('color tint').onChange((value: boolean) => {
      // Update color based on tint setting
      const newColor = (params.useTexture && !value) ? 0xffffff : API.color
      clothMaterial.color.setHex(newColor)
      clothMaterial.needsUpdate = true
    })
    
    materialFolder.add(params, 'fitMode', ['width', 'height']).name('fit mode').onChange(() => {
      // Note: Changing fit mode requires recreating the cloth
      console.log('Fit mode changed. Reload the page to see changes.')
    })
    
    materialFolder
      .addColor(API, 'color')
      .onChange((color: number) => {
        // Only apply color if not using texture, or if color tint is enabled
        if (!params.useTexture || params.useColorTint) {
          clothMaterial.color.setHex(color)
        }
      })
    materialFolder.add(params, 'opacity', 0.0, 1.0, 0.01).name('opacity').onChange((value: number) => {
      clothMaterial.opacity = value
      clothMaterial.needsUpdate = true
    })
    materialFolder.add(clothMaterial, 'roughness', 0.0, 1, 0.01)
    materialFolder.add(clothMaterial, 'sheen', 0.0, 1, 0.01)
    materialFolder.add(clothMaterial, 'sheenRoughness', 0.0, 1, 0.01)
    materialFolder
      .addColor(API, 'sheenColor')
      .onChange((color: number) => {
        clothMaterial.sheenColor.setHex(color)
      })

    // Controls Folder
    const controlsFolder = gui.addFolder('Controls')
    controlsFolder.add(dragControls, 'enabled').name('drag box')
    controlsFolder.add(cameraControls, 'autoRotate').name('auto rotate')
    
    // Box Folder
    const boxFolder = gui.addFolder('Box')
    boxFolder.add(params, 'boxEnabled').name('enable collision')
    boxFolder.add(params, 'boxVisible').name('show mesh')

    // Lights Folder
    const lightsFolder = gui.addFolder('Lights')
    lightsFolder.add(pointLight, 'visible').name('point light')
    lightsFolder.add(ambientLight, 'visible').name('ambient light')

    // Helpers Folder
    const helpersFolder = gui.addFolder('Helpers')
    helpersFolder.add(axesHelper, 'visible').name('axes')
    helpersFolder.add(pointLightHelper, 'visible').name('point light helper')
    helpersFolder.add(gridHelper, 'visible').name('grid')

    // persist GUI state in local storage on changes
    gui.onFinishChange(() => {
      const guiState = gui.save()
      localStorage.setItem('guiState', JSON.stringify(guiState))
    })

    // load GUI state if available in local storage
    const guiState = localStorage.getItem('guiState')
    if (guiState) gui.load(JSON.parse(guiState))

    // reset GUI state button
    const resetGui = () => {
      localStorage.removeItem('guiState')
      gui.reset()
    }
    gui.add({ resetGui }, 'resetGui').name('RESET')

    // gui.close()
  }

  renderer.setAnimationLoop(render)
}

/**
 * ===== 🧵 VERLET GEOMETRY SETUP =====
 * Sets up the geometry of the verlet physics system - a grid of vertices connected by springs.
 * This creates the underlying structure that simulates cloth physics.
 */
function setupVerletGeometry() {
  /**
   * Helper function to add a verlet vertex to the simulation
   * @param x - X position in 3D space
   * @param y - Y position in 3D space
   * @param z - Z position in 3D space
   * @param isFixed - Whether this vertex is fixed in place (won't move)
   */
  const addVerletVertex = (x: number, y: number, z: number, isFixed: boolean) => {
    const id = verletVertices.length
    const vertex = {
      id,
      position: new THREE.Vector3(x, y, z),
      isFixed,
      springIds: [] as number[], // Track which springs are connected to this vertex
    }
    verletVertices.push(vertex)
    return vertex
  }

  /**
   * Helper function to add a spring connecting two vertices
   * Springs create the structural integrity of the cloth
   */
  const addVerletSpring = (vertex0: any, vertex1: any) => {
    const id = verletSprings.length
    const spring = {
      id,
      vertex0,
      vertex1,
    }
    // Register this spring with both vertices
    vertex0.springIds.push(id)
    vertex1.springIds.push(id)
    verletSprings.push(spring)
    return spring
  }

  // Create the cloth's verlet vertices in a grid pattern
  for (let x = 0; x <= clothNumSegmentsX; x++) {
    const column = []
    for (let y = 0; y <= clothNumSegmentsY; y++) {
      // Calculate position in 3D space
      const posX = x * (clothWidth / clothNumSegmentsX) - clothWidth * 0.5
      const posZ = y * (clothHeight / clothNumSegmentsY)
      // Fix all vertices along the top edge - they'll be attached to the anchor box
      const isFixed = y === 0
      const vertex = addVerletVertex(posX, clothHeight * 0.5, posZ, isFixed)
      column.push(vertex)
    }

    verletVertexColumns.push(column)
  }

  // Create the cloth's verlet springs to connect nearby vertices
  // This creates both structural springs (horizontal/vertical) and shear springs (diagonal)
  for (let x = 0; x <= clothNumSegmentsX; x++) {
    for (let y = 0; y <= clothNumSegmentsY; y++) {
      const vertex0 = verletVertexColumns[x][y]
      // Connect to left neighbor
      if (x > 0) addVerletSpring(vertex0, verletVertexColumns[x - 1][y])
      // Connect to top neighbor
      if (y > 0) addVerletSpring(vertex0, verletVertexColumns[x][y - 1])
      // Connect to top-left diagonal (shear spring)
      if (x > 0 && y > 0)
        addVerletSpring(vertex0, verletVertexColumns[x - 1][y - 1])
      // Connect to bottom-left diagonal (shear spring)
      if (x > 0 && y < clothNumSegmentsY)
        addVerletSpring(vertex0, verletVertexColumns[x - 1][y + 1])

      // You can make the cloth more rigid by adding more springs between further apart vertices:
      // if (x > 1) addVerletSpring(vertex0, verletVertexColumns[x - 2][y]);
      // if (y > 1) addVerletSpring(vertex0, verletVertexColumns[x][y - 2]);
    }
  }
}

/**
 * ===== 📊 VERLET VERTEX BUFFERS SETUP =====
 * Creates GPU buffers to hold vertex data for the compute shaders.
 * These buffers allow the physics simulation to run on the GPU for better performance.
 */
function setupVerletVertexBuffers() {
  const vertexCount = verletVertices.length

  // This array will hold a list of spring IDs, ordered by the vertex they affect
  // This allows the compute shader to efficiently iterate over all springs affecting each vertex
  const springListArray: number[] = []

  // Position buffer: stores xyz coordinates for each vertex
  const vertexPositionArray = new Float32Array(vertexCount * 3)

  // Parameters buffer: stores three values for each vertex:
  // x: isFixed (1 if immovable, 0 if movable)
  // y: springCount (number of springs connected to this vertex)
  // z: springPointer (index of first spring in springListArray)
  const vertexParamsArray = new Uint32Array(vertexCount * 3)

  // Populate the arrays with vertex data
  for (let i = 0; i < vertexCount; i++) {
    const vertex = verletVertices[i]

    // Store initial position
    vertexPositionArray[i * 3] = vertex.position.x
    vertexPositionArray[i * 3 + 1] = vertex.position.y
    vertexPositionArray[i * 3 + 2] = vertex.position.z

    // Store whether this vertex is fixed
    vertexParamsArray[i * 3] = vertex.isFixed ? 1 : 0

    // For movable vertices, store spring connection info
    if (!vertex.isFixed) {
      vertexParamsArray[i * 3 + 1] = vertex.springIds.length
      vertexParamsArray[i * 3 + 2] = springListArray.length
      springListArray.push(...vertex.springIds)
    }
  }

  // Create GPU buffers from the arrays
  // setPBO(true) enables Pixel Buffer Objects for WebGL fallback compatibility
  vertexPositionBuffer = instancedArray(vertexPositionArray, 'vec3').setPBO(true)
  vertexForceBuffer = instancedArray(vertexCount, 'vec3') // Stores accumulated forces
  vertexParamsBuffer = instancedArray(vertexParamsArray, 'uvec3')

  springListBuffer = instancedArray(
    new Uint32Array(springListArray),
    'uint'
  ).setPBO(true)
}

function setupVerletSpringBuffers() {
  // setup the buffers holding the spring data for the compute shaders

  const springCount = verletSprings.length

  const springVertexIdArray = new Uint32Array(springCount * 2)
  const springRestLengthArray = new Float32Array(springCount)

  for (let i = 0; i < springCount; i++) {
    const spring = verletSprings[i]
    springVertexIdArray[i * 2] = spring.vertex0.id
    springVertexIdArray[i * 2 + 1] = spring.vertex1.id
    springRestLengthArray[i] = spring.vertex0.position.distanceTo(
      spring.vertex1.position
    )
  }

  springVertexIdBuffer = instancedArray(springVertexIdArray, 'uvec2').setPBO(true)
  springRestLengthBuffer = instancedArray(springRestLengthArray, 'float')
  springForceBuffer = instancedArray(springCount * 3, 'vec3').setPBO(true)
}

function setupUniforms() {
  dampeningUniform = uniform(params.dampening)
  boxPositionUniform = uniform(new THREE.Vector3(0, 0, 0))
  boxUniform = uniform(1.0)
  windUniform = uniform(1.0)
  stiffnessUniform = uniform(0.2)
  boxSizeUniform = uniform(new THREE.Vector3(boxWidth / 2, boxHeight / 2, boxDepth / 2))
  anchorPositionUniform = uniform(new THREE.Vector3(0, clothHeight * 0.5, 0))
  gravityUniform = uniform(params.gravity)
}

/**
 * ===== 🖥️ COMPUTE SHADERS SETUP =====
 * Sets up GPU compute shaders for the Verlet physics simulation.
 * These shaders run on the GPU for high-performance parallel computation.
 * Two shaders are executed for each simulation step:
 * 1. Spring Forces - calculates tension in each spring
 * 2. Vertex Forces - accumulates all forces and updates vertex positions
 */
function setupComputeShaders() {
  const vertexCount = verletVertices.length
  const springCount = verletSprings.length

  /**
   * COMPUTE SHADER 1: Spring Forces
   * Calculates the force exerted by each spring based on Hooke's Law:
   * F = k * (current_length - rest_length)
   * where k is the stiffness constant
   */
  computeSpringForces = Fn(() => {
    // Compute shaders are executed in groups of 64, so instanceIndex might exceed the spring count
    If(instanceIndex.greaterThanEqual(uint(springCount)), () => {
      Return()
    })

    // Get the two vertices connected by this spring
    const vertexIds = springVertexIdBuffer.element(instanceIndex)
    const restLength = springRestLengthBuffer.element(instanceIndex)

    const vertex0Position = vertexPositionBuffer.element(vertexIds.x)
    const vertex1Position = vertexPositionBuffer.element(vertexIds.y)

    // Calculate spring force using Hooke's Law
    const delta = vertex1Position.sub(vertex0Position).toVar()
    const dist = delta.length().max(0.000001).toVar() // Prevent division by zero
    const force = dist
      .sub(restLength) // How much the spring is stretched/compressed
      .mul(stiffnessUniform) // Multiply by stiffness constant
      .mul(delta) // Direction of the force
      .mul(0.5) // Split force between both vertices
      .div(dist) // Normalize
    springForceBuffer.element(instanceIndex).assign(force)
  })()
    .compute(springCount) as any

  /**
   * COMPUTE SHADER 2: Vertex Forces
   * Accumulates all forces acting on each vertex and updates its position.
   * Forces include:
   * - Spring tensions from connected springs
   * - Gravity (constant downward force)
   * - Wind (using 3D noise for realistic turbulence)
   * - Box collision (repulsion when vertex penetrates the box)
   */
  computeVertexForces = Fn(() => {
    // Early exit for indices beyond vertex count
    If(instanceIndex.greaterThanEqual(uint(vertexCount)), () => {
      Return()
    })

    // Load vertex parameters
    const params = vertexParamsBuffer.element(instanceIndex).toVar()
    const isFixed = params.x // Whether this vertex is fixed in place
    const springCount = params.y // Number of springs connected to this vertex
    const springPointer = params.z // Index into spring list

    // For fixed vertices (top edge), update position to follow anchor box
    If(isFixed, () => {
      const currentPos = vertexPositionBuffer.element(instanceIndex).toVar()
      // Keep the X position relative to anchor, update Y and Z to match anchor
      const newPos = currentPos.toVar()
      newPos.y = anchorPositionUniform.y
      newPos.z = anchorPositionUniform.z
      vertexPositionBuffer.element(instanceIndex).assign(newPos)
      Return()
    })

    const position = vertexPositionBuffer
      .element(instanceIndex)
      .toVar('vertexPosition')
    const force = vertexForceBuffer.element(instanceIndex).toVar('vertexForce')

    // Apply dampening to simulate air resistance and energy loss
    force.mulAssign(dampeningUniform)

    // Accumulate forces from all connected springs
    const ptrStart = springPointer.toVar('ptrStart')
    const ptrEnd = ptrStart.add(springCount).toVar('ptrEnd')

    Loop({ start: ptrStart, end: ptrEnd, type: 'uint', condition: '<' }, ({ i }) => {
      const springId = springListBuffer.element(i).toVar('springId')
      const springForce = springForceBuffer.element(springId)
      const springVertexIds = springVertexIdBuffer.element(springId)
      // Determine if we should add or subtract the force based on which end of the spring we are
      const factor = select(springVertexIds.x.equal(instanceIndex), 1.0, -1.0)
      force.addAssign(springForce.mul(factor))
    })

    // Add gravity (downward force controlled by gravity uniform)
    force.y.subAssign(gravityUniform)

    // Add wind force using 3D Perlin noise for realistic turbulence
    const noise = triNoise3D(position, 1, time).sub(0.2).mul(0.0001)
    const windForce = noise.mul(windUniform)
    force.z.subAssign(windForce)

    // Handle collision with box - smooth approach similar to sphere
    const nextPosition = position.add(force)
    const deltaBox = nextPosition.sub(boxPositionUniform)
    const halfSize = boxSizeUniform
    
    // Clamp to find closest point on box surface
    const closestX = deltaBox.x.clamp(halfSize.x.negate(), halfSize.x)
    const closestY = deltaBox.y.clamp(halfSize.y.negate(), halfSize.y)
    const closestZ = deltaBox.z.clamp(halfSize.z.negate(), halfSize.z)
    
    // Vector from closest point to vertex
    const diffX = deltaBox.x.sub(closestX)
    const diffY = deltaBox.y.sub(closestY)
    const diffZ = deltaBox.z.sub(closestZ)
    
    // Distance from vertex to closest point on box
    const distSq = diffX.mul(diffX).add(diffY.mul(diffY)).add(diffZ.mul(diffZ))
    const dist = distSq.sqrt().max(0.0001)
    
    // Smooth repulsion force (similar to sphere collision)
    // Force increases as vertex gets closer to box
    const repulsionRadius = float(0.4) // Repulsion starts this distance from box surface
    const penetration = repulsionRadius.sub(dist).max(0)
    
    const boxForceDir = float(0).toVar()
    boxForceDir.x = diffX.div(dist)
    boxForceDir.y = diffY.div(dist)
    boxForceDir.z = diffZ.div(dist)
    
    // Increased force multiplier for more dramatic interaction with heavy cloth
    const boxForce = boxForceDir.mul(penetration).mul(0.01).mul(boxUniform)
    force.addAssign(boxForce)

    // Update the force and position buffers
    vertexForceBuffer.element(instanceIndex).assign(force)
    vertexPositionBuffer.element(instanceIndex).addAssign(force)
  })()
    .compute(vertexCount) as any
}

function setupWireframe() {
  // adds helpers to visualize the verlet system

  // verlet vertex visualizer
  const vertexWireframeMaterial = new THREE.SpriteNodeMaterial()
  vertexWireframeMaterial.positionNode = vertexPositionBuffer.element(instanceIndex)
  vertexWireframeObject = new THREE.Mesh(
    new THREE.PlaneGeometry(0.01, 0.01),
    vertexWireframeMaterial
  )
  vertexWireframeObject.frustumCulled = false
  vertexWireframeObject.count = verletVertices.length
  scene.add(vertexWireframeObject)

  // verlet spring visualizer
  const springWireframePositionBuffer = new THREE.BufferAttribute(
    new Float32Array(6),
    3,
    false
  )
  const springWireframeIndexBuffer = new THREE.BufferAttribute(
    new Uint32Array([0, 1]),
    1,
    false
  )
  const springWireframeMaterial = new THREE.LineBasicNodeMaterial()
  springWireframeMaterial.positionNode = Fn(() => {
    const vertexIds = springVertexIdBuffer.element(instanceIndex)
    const vertexId = select(
      attribute('vertexIndex').equal(0),
      vertexIds.x,
      vertexIds.y
    )
    return vertexPositionBuffer.element(vertexId)
  })()

  const springWireframeGeometry = new THREE.InstancedBufferGeometry()
  springWireframeGeometry.setAttribute('position', springWireframePositionBuffer)
  springWireframeGeometry.setAttribute('vertexIndex', springWireframeIndexBuffer)
  springWireframeGeometry.instanceCount = verletSprings.length

  springWireframeObject = new THREE.Line(
    springWireframeGeometry,
    springWireframeMaterial
  )
  springWireframeObject.frustumCulled = false
  springWireframeObject.count = verletSprings.length
  scene.add(springWireframeObject)
}

function setupBox() {
  const geometry = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth)
  const material = new THREE.MeshStandardNodeMaterial()
  box = new THREE.Mesh(geometry, material)
  scene.add(box)
}

function setupAnchorBox() {
  // Create the anchor box that the cloth will be attached to
  const geometry = new THREE.BoxGeometry(clothWidth, 0.1, 0.2)
  const material = new THREE.MeshStandardNodeMaterial({
    color: 0x808080,
    metalness: 0.5,
    roughness: 0.5,
  })
  anchorBox = new THREE.Mesh(geometry, material)
  anchorBox.position.set(0, clothHeight * 0.5, params.anchorZ)
  anchorBox.castShadow = true
  anchorBox.receiveShadow = true
  scene.add(anchorBox)
}

function setupClothMesh() {
  // This function generates a three Geometry and Mesh to render the cloth based on the verlet systems position data.
  // Therefore it creates a plane mesh, in which each vertex will be centered in the center of 4 verlet vertices.

  const vertexCount = clothNumSegmentsX * clothNumSegmentsY
  const geometry = new THREE.BufferGeometry()

  // verletVertexIdArray will hold the 4 verlet vertex ids that contribute to each geometry vertex's position
  const verletVertexIdArray = new Uint32Array(vertexCount * 4)
  const indices: number[] = []

  const getIndex = (x: number, y: number) => {
    return y * clothNumSegmentsX + x
  }

  for (let x = 0; x < clothNumSegmentsX; x++) {
    for (let y = 0; y < clothNumSegmentsX; y++) {
      const index = getIndex(x, y)
      verletVertexIdArray[index * 4] = verletVertexColumns[x][y].id
      verletVertexIdArray[index * 4 + 1] = verletVertexColumns[x + 1][y].id
      verletVertexIdArray[index * 4 + 2] = verletVertexColumns[x][y + 1].id
      verletVertexIdArray[index * 4 + 3] = verletVertexColumns[x + 1][y + 1].id

      if (x > 0 && y > 0) {
        indices.push(
          getIndex(x, y),
          getIndex(x - 1, y),
          getIndex(x - 1, y - 1)
        )
        indices.push(
          getIndex(x, y),
          getIndex(x - 1, y - 1),
          getIndex(x, y - 1)
        )
      }
    }
  }

  const verletVertexIdBuffer = new THREE.BufferAttribute(
    verletVertexIdArray,
    4,
    false
  )
  const positionBuffer = new THREE.BufferAttribute(
    new Float32Array(vertexCount * 3),
    3,
    false
  )
  
  // Create UV coordinates for texture mapping
  const uvArray = new Float32Array(vertexCount * 2)
  for (let x = 0; x < clothNumSegmentsX; x++) {
    for (let y = 0; y < clothNumSegmentsY; y++) {
      const index = getIndex(x, y)
      // Map UVs from 0 to 1 across the cloth
      // U coordinate: flip horizontally (1 to 0)
      uvArray[index * 2] = 1 - (x / (clothNumSegmentsX - 1))
      // V coordinate: flip to show image right-side up (1 to 0)
      uvArray[index * 2 + 1] = 1 - (y / (clothNumSegmentsY - 1))
    }
  }
  const uvBuffer = new THREE.BufferAttribute(uvArray, 2, false)
  
  geometry.setAttribute('position', positionBuffer)
  geometry.setAttribute('vertexIds', verletVertexIdBuffer)
  geometry.setAttribute('uv', uvBuffer)
  geometry.setIndex(indices)

  // Set color to white if using texture without tinting, otherwise use API color
  const materialColor = (params.useTexture && !params.useColorTint) 
    ? 0xffffff 
    : API.color
  
  clothMaterial = new THREE.MeshPhysicalNodeMaterial({
    color: new THREE.Color().setHex(materialColor),
    map: params.useTexture ? clothTexture : null,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: params.opacity,
    sheen: 1.0,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color().setHex(API.sheenColor),
  })

  clothMaterial.positionNode = Fn(({ material }) => {
    // gather the position of the 4 verlet vertices and calculate the center position and normal from that
    const vertexIds = attribute('vertexIds')
    const v0 = vertexPositionBuffer.element(vertexIds.x).toVar()
    const v1 = vertexPositionBuffer.element(vertexIds.y).toVar()
    const v2 = vertexPositionBuffer.element(vertexIds.z).toVar()
    const v3 = vertexPositionBuffer.element(vertexIds.w).toVar()

    const top = v0.add(v1)
    const right = v1.add(v3)
    const bottom = v2.add(v3)
    const left = v0.add(v2)

    const tangent = right.sub(left).normalize()
    const bitangent = bottom.sub(top).normalize()

    const normal = cross(tangent, bitangent)

    // send the normalView from the vertex shader to the fragment shader
    ;(material as any).normalNode = transformNormalToView(normal).toVarying()

    return v0.add(v1).add(v2).add(v3).mul(0.25)
  })()

  clothMesh = new THREE.Mesh(geometry, clothMaterial)
  clothMesh.frustumCulled = false
  scene.add(clothMesh)
}

function setupCloth() {
  setupVerletGeometry()
  setupVerletVertexBuffers()
  setupVerletSpringBuffers()
  setupUniforms()
  setupComputeShaders()
  setupWireframe()
  setupBox()
  setupAnchorBox()
  setupClothMesh()
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
}

function updateBox() {
  // Simple back and forth motion along Z axis, positioned lower
  box.position.set(
    0, 
    (-boxHeight / 2) - 0.1, // Move down by half the box height
    Math.sin(timestamp * 0.8) * 0.5 // Back and forth along Z
  )
  boxPositionUniform.value.copy(box.position)
}

function updateAnchorBox() {
  // Update anchor box position based on GUI parameter
  anchorBox.position.z = params.anchorZ
  
  // Update the uniform that the compute shader uses to position fixed vertices
  anchorPositionUniform.value.set(0, clothHeight * 0.5, params.anchorZ)
}

async function render() {
  stats.begin()

  // Update visibility and physics based on GUI parameters
  box.visible = params.boxVisible // Controls mesh visibility
  boxUniform.value = params.boxEnabled ? 1 : 0 // Controls collision physics
  windUniform.value = params.wind
  dampeningUniform.value = params.dampening
  gravityUniform.value = params.gravity
  clothMesh.visible = !params.wireframe
  vertexWireframeObject.visible = params.wireframe
  springWireframeObject.visible = params.wireframe

  // Handle window resize
  if (resizeRendererToDisplaySize(renderer)) {
    const canvas = renderer.domElement
    camera.aspect = canvas.clientWidth / canvas.clientHeight
    camera.updateProjectionMatrix()
  }

  // Update camera controls
  cameraControls.update()

  // Run physics simulation
  const deltaTime = Math.min(clock.getDelta(), 1 / 60) // don't advance the time too far, for example when the window is out of focus
  const stepsPerSecond = 360 // ensure the same amount of simulation steps per second on all systems, independent of refresh rate
  const timePerStep = 1 / stepsPerSecond

  timeSinceLastStep += deltaTime

  while (timeSinceLastStep >= timePerStep) {
    // run a verlet system simulation step
    timestamp += timePerStep
    timeSinceLastStep -= timePerStep
    updateBox()
    updateAnchorBox()
    renderer.compute(computeSpringForces)
    renderer.compute(computeVertexForces)
  }

  renderer.render(scene, camera)
  stats.end()
}
