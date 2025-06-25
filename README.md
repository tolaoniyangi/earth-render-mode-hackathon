# Earth Canvas

## The Problem
We generate design options, as simple block massing, on Earth. While these are placed within their real-world context, visualizing those designs in a realiztic manner is slow and expensive. Traditional rendering takes days and needs specialized skills.

## Our Solution
Studio Mode for Earth rapidly transforms simple 3D massing models into photorealistic images directly within Google Earth. We cut out the rendering bottleneck by leveraging AI.

## How It Works
We bridge conceptual 3D models with photorealism by integrating ComfyUI and geospatial data:

**Massing Input**: Provide a basic 3D model (e.g., KML/KMZ).

**Google Earth Context**: We capture the surrounding environment (terrain, buildings, roads) from Google Earth.

**AI Photorealism (ComfyUI)**: A custom ComfyUI workflow, using advanced Stable Diffusion models, intelligently fills in details, textures, and lighting, transforming massing into a detailed, context-aware render.

**Google Earth Integration**: The photorealistic image is seamlessly overlaid back into Google Earth for immediate visualization.

## Technologies
**ComfyUI**: AI rendering engine for Stable Diffusion workflows.

**Stable Diffusion Models**: Specialized for architectural/geospatial photorealism.

**Python**: For scripting and workflow orchestration.

Eventually, we could use Google Earth for data interaction and display. For now, we're just using screenshots.

## Why It Matters
We're solving a customer pain point of making generated designs presentation-ready. This project makes  rendering accessible for contextual architectural and urban photorealism in a simple, user-manner. We bridge readily available geospatial data with advanced generative AI to create geolocated, contextually accurate renders, not just generic images.