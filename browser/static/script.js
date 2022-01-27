/* global d3, nv, event_impact */

function appendFilterEmboss(defs, name, color = 'white', scale = 1) {
  const filter = defs.append('filter')
    .attr('id', name)
    .attr('filterUnits', 'objectBoundingBox')
    .attr('x', '0%')
    .attr('y', '0%')
    .attr('width', '100%')
    .attr('height', '100%')

  filter.append('feOffset').attr({
    dx: '0',
    dy: '0',
    in: 'SourceGraphic',
    // result: 'A',
  })
  // filter.append('feOffset').attr({
  //   dx: '0',
  //   dy: '10',
  //   in: 'SourceGraphic',
  //   result: 'B',
  // })

  // filter.append('feComposite').attr({
  //   in: 'B',
  //   in2: 'A',
  //   operator: 'over',
  // })

  filter.append('feComponentTransfer').html(`
    <feFuncR type="discrete" tableValues="0 1"/>
    <feFuncG type="discrete" tableValues="0 1"/>
    <feFuncB type="discrete" tableValues="0 1"/>
    <feFuncA type="discrete" tableValues="0 1"/>
  `)

  filter.append('feGaussianBlur').attr({
    class: 'gb1',
    stdDeviation: 5,
    result: 'SourceGraphic_blurred',
  })

  filter.append('feDiffuseLighting').attr({
    in: 'SourceGraphic_blurred',
    'lighting-color': color,
    // 'lighting-color': 'white',
    surfaceScale: 60,
  }).append('feDistantLight').attr({
    azimuth: 45,
    // elevation: -200,
    // elevation: 480,
    elevation: -210,
  })

  filter.append('feGaussianBlur').attr({
    class: 'gb2',
    stdDeviation: 3,
  })

  filter.append('feComponentTransfer')
    .attr('result', 'OUTPUT')
    .append('feFuncA')
    .attr({ type: 'linear', slope: '0', intercept: '1' })

  // SVG threshold
  filter.append('feComponentTransfer')
    .attr('in', 'SourceGraphic_blurred')
    .attr('color-interpolation-filters', 'sRGB')
    .attr('result', 'SourceGraphic_blob')
    .html(`
      <feFuncR type="discrete" tableValues="0 1"/>
      <feFuncG type="discrete" tableValues="0 1"/>
      <feFuncB type="discrete" tableValues="0 1"/>
      <feFuncA type="discrete" tableValues="0 1"/>
  `)


  filter.append('feComposite')
    .attr('in', 'OUTPUT')
    .attr('operator', 'in')
    .attr('in2', 'SourceGraphic_blurred')

  filter.append('feComponentTransfer')
    .append('feFuncA')
    .attr({ type: 'linear', slope: '0', intercept: '1' })

  filter.append('feComposite')
    .attr('operator', 'in')
    .attr('in2', 'SourceGraphic')

  // filter.append('feComposite')
  //   .attr('in', 'OUTPUT')
  //   .attr('operator', 'in')
  //   .attr('in2', 'SourceGraphic_blob')


  // filter.append('feFlood')
  //   .attr('result', 'floodFill')
  //   .attr('x', '0')
  //   .attr('y', '0')
  //   .attr('width', '100%')
  //   .attr('height', '100%')
  //   .attr('flood-color', 'red')
  //   .attr('flood-opacity', '1')
  // filter.append('feBlend')
  //   .attr('in', 'composite')
  //   .attr('in2', 'floodFill')
  //   .attr('mode', 'multiply')
  // filter.append('feImage')
  //   .attr('xlink:href', 'data:image/svg+xml;charset=utf-8,<svg width="100%" height="100%"><rect fill="yellow" width="100%" height="100%" /></svg>')
  // filter.append('feComposite')
  //   .attr('in', 'SourceGraphic')
  //   .attr('operator', 'arithmetic')
  //   .attr('in2', 'lorem')
  //   .attr('k1', '1')
  //   .attr('k2', '0.1')
  //   .attr('k3', '0.0')
  //   .attr('k4', '-0.1')
}

function estimateTextBBox(text, attrs, styles) {
  const svg = d3.select('#chart1').append('svg')
  const textNode = svg.append('text')
    .attr('x', 0)
    .attr('y', 0)
    .text(text)
  textNode.attr(attrs)
  textNode.style(styles)

  const estimateTextBBox = textNode.node().getBBox()
  svg.remove()
  return estimateTextBBox
}

// make text pattern
function textPatternFactory(defs, id, { text, fg, bg, strokeColor = 'none', angle = 0, lineHeight = 1.6 }) {
  const textAttrs = {
    'font-size': 15,
    'font-family': 'sans-serif',
    // 'text-anchor': 'middle',
    // 'alignment-baseline': 'middle',
    'dominant-baseline': 'central',
    'line-height': lineHeight,
  }

  const textBBox = estimateTextBBox(text, textAttrs, {})

  const x0 = textBBox.x
  const y0 = textBBox.y
  const dh = textBBox.height * textAttrs['line-height']
  const dw = textBBox.width + textAttrs['font-size'] * 0.95

  const pattern = defs
    .append('pattern')
    .attr('id', id)
    .attr('patternUnits', 'userSpaceOnUse')
    .attr('patternTransform', `rotate(${angle})`)
    .attr('height', dh * (10 - 1))
    .attr('width', dw * 2)

  pattern.append('rect').attr({
    height: '100%',
    width: '100%',
    fill: bg,
  })

  for (let x = -1; x < 3; x++) {
    for (let y = 0; y < 10; y++) {
      pattern.append('text')
        .attr('x', x0 + x * dw + Math.cos(y) * dw * 0.3)
        .attr('y', y0 + y * dh + 0 * Math.cos(x) * dh * 0.1)
        .style('fill', fg)
        .attr(textAttrs)
        .attr('stroke', strokeColor)
        .attr('stroke-width', '1px')
        .style('paint-order', 'stroke')
        .text(text)
    }
  }
  return pattern
}

function slugifyToId(str) {
  return str.toLowerCase().replace(/[!-/:-@\s]+/g, '-')
}

function chartStyleLines(svg, event_impact, config) {
  const offsetY = d3.scale.ordinal()
    .domain(d3.range(event_impact.length))
    .rangeRoundBands([0, config.height], 1)
  const offsetYDelta = offsetY(1) - offsetY(0)

  const minmax = (arr) => {
    const min = Math.min(...arr)
    const max = Math.max(...arr)
    return [min, max]
  }
  const mM = minmax(event_impact.flatMap(d => d.values.map(d => d[1])))
  const scaleY = d3.scale.linear()
    .domain(mM)
    .range([0, 1])

  const getPoints = (singleEventImpact) => {
    // make time range
    const rangeTime = d3.extent(singleEventImpact.values, d => d[0])
    // const rangeValue = d3.extent(singleEventImpact.values, d => d[1]);

    const arrTop = []
    const arrBottom = []

    const scaleX = d3.scale.linear()
      .domain(rangeTime)
      .range([0, config.width])

    const eventOffset = offsetY(singleEventImpact.seriesIndex)
    // const N = singleEventImpact.values.length
    singleEventImpact.values.forEach((d) => {
      const [time, value] = d
      const dx = scaleX(time)
      const dy = eventOffset
      const offset = scaleY(value) * offsetYDelta
      arrTop.push([dx, dy - offset * config.scaleHeightBottom])
      arrBottom.push([dx, dy + offset * config.scaleHeightTop])
    })
    const arr = arrTop.concat(arrBottom.reverse())
    return arr.map(x => x.join(',')).join(' ')
  }


  svg
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('viewBox', `0 0 ${config.width} ${config.height}`)
    .style('width', '100%')
    .style('height', CSS.percent(100 * config.height / config.width))
    .style('background', 'white')
    // .style('background', 'linear-gradient(to bottom left, #0ff, #606, #00f)')

  // const bg = svg
  //   .selectAll('g.background')
  //   .data(event_impact)
  //   .enter()
  //   .append('g')
  //   .attr('class', 'background')

  // const hh = (d) => (config.scaleHeightTop + config.scaleHeightBottom) * offsetYDelta

  // bg.append('rect').attr({
  //   x: 0,
  //   y: d => {
  //     const h = hh(d)
  //     const y = offsetY(d.seriesIndex)
  //     return y - h / 2
  //   },
  //   width: config.width,
  //   height: hh,
  //   fill: d => {
  //     const c = d3.lab(d.color)
  //     c.l = 90
  //     return c
  //   },
  // })


  // const fg = svg
  //   .selectAll('g.series')
  //   .data(event_impact)
  //   .enter()
  //   .append('g')
  //   .attr('class', 'series')

  svg.selectAll('g.series').remove()

  const fg = svg
    .selectAll('g.series')
    .data(event_impact)
    .enter()
    .append('g')
    .attr('class', 'series')

  fg.append('polygon')
    .attr('fill', '#fff')
    // .attr('fill', (d) => d.color)
    .attr('points', getPoints)
    .attr('filter', d => `url("#emboss-${slugifyToId(d.key)}")`)

  fg.append('polygon')
    // .attr('fill', 'grey')
    .attr('fill', (d) => `url("#pattern-textonly--${slugifyToId(d.key)}")`)
    // .attr('stroke', (d) => d.color)
    .attr('points', getPoints)
}

let chart;

function init(event_impact) {
  event_impact.sort((a, b) => {
    // sort by max .values
    const maxA = d3.max(a.values, d => d[1])
    const maxB = d3.max(b.values, d => d[1])
    return maxA - maxB
  })

  const colors = d3.scale.category20()

  nv.addGraph(function() {
    chart = nv.models.stackedAreaChart()

    chart
      .useInteractiveGuideline(true)
      .x((d) => d[0])
      .y((d) => d[1])
      .controlLabels({ stacked: 'Stacked' })
      .showControls(true)
      .clipEdge(true)
      .duration(500)

    chart.yAxis.scale().domain([0, 20])
    chart.yAxis.tickFormat(d3.format(',.1f'))

    chart.xAxis.tickFormat(function(d) { return d3.time.format('%x')(new Date(d)) })

    chart.height(700)

    chart.interpolate('basis')
    // linear - piecewise linear segments, as in a polyline.
    // linear-closed - close the linear segments to form a polygon.
    // step-before - alternate between vertical and horizontal segments, as in a step function.
    // step-after - alternate between horizontal and vertical segments, as in a step function.
    // basis - a B-spline, with control point duplication on the ends.
    // basis-open - an open B-spline; may not intersect the start or end.
    // basis-closed - a closed B-spline, as in a loop.
    // bundle - equivalent to basis, except the tension parameter is used to straighten the spline.
    // cardinal - a Cardinal spline, with control point duplication on the ends.
    // cardinal-open - an open Cardinal spline; may not intersect the start or end, but will intersect other control points.
    // cardinal-closed - a closed Cardinal spline, as in a loop.
    // monotone - cubic interpolation that preserves monotonicity in y.


    chart.legend.vers('furious')

    chart.showControls(false)

    nv.utils.windowResize(chart.update)

    update(event_impact)

    return chart
  })

  const event_table = document.querySelector('#events')
  event_table.addEventListener('click', function (e) {
    if (e.target.tagName === 'TD') {
      const tr = e.target.parentNode
      const tds = tr.children
      const start = new Date(tds[1].innerText)
      const end = new Date(tds[2].innerText)
      chart.xDomain([start, end]).update()
    }
  })

  document.querySelector('#event_impact').addEventListener('click', function () {
    const rows = event_table.querySelectorAll('tr')
    let min_date = +Infinity
    let max_date = -Infinity
    // let best_score = 0

    for (const row of rows) {
      const tds = row.children
      const start = new Date(tds[1].innerText)
      const end = new Date(tds[2].innerText)

      if (start < min_date) {
        min_date = start
      }
      if (end > max_date) {
        max_date = end
      }
    }

    min_date = new Date((+min_date) - 24 * 3600 * 1000)
    max_date = new Date((+max_date) + 24 * 3600 * 1000)

    chart.xDomain([min_date, max_date]).update()
  })

  // window.addEventListener('mousemove', (e) => {
  //   const mouseX = e.clientX
  //   const mouseXPercent = mouseX / window.innerWidth
  //   const mouseY = e.clientY
  //   const mouseYPercent = mouseY / window.innerHeight

  //   if (document.querySelector('filter[id^="emboss"]')) {
  //     document.querySelectorAll('filter[id^="emboss-"]').forEach((x) => {
  //       const feDiffuseLighting = x.querySelector('feDiffuseLighting')
  //       const feDistantLight = x.querySelector('feDistantLight')
  //       const feGaussianBlur1 = x.querySelector('feGaussianBlur.gb1')
  //       const feGaussianBlur2 = x.querySelector('feGaussianBlur.gb2')
  //       const feOffset = x.querySelector('feOffset')

  //       // feDiffuseLighting?.setAttribute('surfaceScale', mouseXPercent * 500)
  //       // feDistantLight?.setAttribute('azimuth', mouseXPercent * 360)
  //       // feDistantLight?.setAttribute('elevation', (mouseXPercent-0.5) * 1000)

  //       feGaussianBlur1?.setAttribute('stdDeviation', mouseYPercent * 10)
  //       feGaussianBlur2?.setAttribute('stdDeviation', mouseXPercent * 10)

  //       // feOffset?.setAttribute('dy', mouseYPercent * 100)
  //     })
  //   }
  // })

  const form = document.createElement('form')
  form.innerHTML = `
    <style>
      form { font-size: 18px; background: black; color: white; padding: 4em; text-align: left; }
      form label { display: block; }
      form > div { width: fit-content; margin: auto; }
      .spinner-container { display: block; }
      .spinner {
        display: inline-block;
        width: 1em; height: 1em;
        border-radius: 1em;
        border-left: 2px solid currentColor; border-right: 2px solid currentColor;
        animation: spin 1s linear infinite;
      }
      @keyframes spin { 100% { transform: rotate(360deg); } }
      .spinner-container[hidden] { opacity: 0; visibility: hidden; }
    </style>
    <div>
      <label>T = <input name="tsl" type="number" value="1440" step=1 min=1 /> (time slice length)</label>
      <label>maf = <input name="maf" type="number" value="10" step=1 min=1 /> (min. abs. freq.)</label>
      <label>mrf = <input name="mrf" type="number" value="0.4" step=0.1 min=0 max=1 /> (max. rel. freq.)</label>

      <label>K = <input name="k" type="number" value="10" step=1 min=1 /> (number of top events to detect)</label>
      <label>P = <input name="p" type="number" value="10" step=1 min=1 /> (number of candidate words per event)</label>
      <label>θ = <input name="t" type="number" value="0.6" step=0.1 min=0 max=1 /></label>
      <label>σ = <input name="s" type="number" value="0.6" step=0.1 min=0 max=1 /></label>

      <input type="submit" value="Submit" />

      <div class="spinner-container" hidden="hidden">
        Chargement en cours...
        &nbsp;
        <div class="spinner"></div>
      </div>
    </div>
  `
  document.body.appendChild(form)
  form.addEventListener('submit', async function (e) {
    // prevent submit
    e.preventDefault()

    const values = {}
    for (const input of form.querySelectorAll('input')) {
      values[input.name] = input.value
    }

    const spinner = form.querySelector('.spinner-container')
    spinner.removeAttribute('hidden')
    await fetch_update(values)
    spinner.setAttribute('hidden', 'true')
  })
}

function update(event_impact) {
    const svg = d3.select('#chart1')

    let defs = svg.select('defs.all-defs')
    if (!defs.node()) {
      defs = svg.append('defs').attr('class', 'all-defs')
    } else {
      defs.html('') // empty
    }

    d3.select('#chart1')
      .datum(event_impact)
      .call(chart)

    const nvAreas = d3.selectAll('#chart1 .nv-area')
    nvAreas.attr('custom-fill', (d) => slugifyToId(d.key))

    let styleElement = d3.select('#chart1 style')
    if (!styleElement.node()) {
      styleElement = d3.select('#chart1').append('style')
    }

    let css = ''
    nvAreas.each((nvArea) => {
      // console.log(nvArea)
      const index = nvArea.seriesIndex
      const text = nvArea.key

      // const id = index
      const id = slugifyToId(text)

      // if nvArea.color is a color
      let bg
      if (nvArea.color.indexOf('#') === 0) {
        bg = d3.hsl(nvArea.color)
      } else {
        bg = d3.hsl(colors(index))
      }
      const angle = -45 * bg.h / 360
      const fg = bg.l > 0.5 ? bg.darker(4) : bg.brighter(2)
      // const fg = bg.l > 0.5 ? "#000" : "#fff";
      const fgHover = bg.l > 0.5 ? bg.darker(1) : bg.brighter(1)

      textPatternFactory(defs, `pattern--${id}`, {
        text, fg, bg, angle,
      })
      // textPatternFactory(defs, `pattern-hover--${id}`, {
      //   text, bg, angle,
      //   fg: fgHover,
      // })
      // textPatternFactory(defs, `pattern-noangle--${id}`, {
      //   text, bg,
      //   fg: bg.l > 0.5 ? '#000' : '#fff',
      //   angle: 0,
      //   lineHeight: 1.1,
      // })
      textPatternFactory(defs, `pattern-textonly--${id}`, {
        text, bg: 'transparent',
        fg: bg.l > 0.5 ? '#000' : '#fff',
        // strokeColor: bg.l < 0.5 ? '#000' : '#fff',
        angle: 0,
        lineHeight: 1.1,
      })

      appendFilterEmboss(defs, `emboss-${id}`, bg)

      // css += `.nv-area-${index} { fill: url("#pattern--${id}") !important; }` + '\n'
      // css += `.nv-area-${index}:hover { fill: url("#pattern-hover--${id}") !important; }` + '\n'
      css += `[custom-fill="${id}"] { fill: url("#pattern--${id}") !important; }` + '\n'
      // css += `[custom-fill="${id}"]:hover { fill: url("#pattern-hover--${id}") !important; }` + '\n'
    })
    styleElement.text(css)

    const chartStyleLinesParams1 = {
      width: 1400,
      height: 1000,
      scaleHeightTop: 0, // 0.5 to not have distorsion
      scaleHeightBottom: 2, // 0.5 to not have distorsion
    }
    const chartStyleLinesParams2 = {
      width: 1400,
      height: 1000,
      scaleHeightTop: 0.5, // 0.5 to not have distorsion
      scaleHeightBottom: 0.5, // 0.5 to not have distorsion
    }

    chartStyleLines(d3.select('#chart-style-lines'), event_impact, chartStyleLinesParams1)
    chartStyleLines(d3.select('#chart-style-lines-mirror'), event_impact, chartStyleLinesParams2)

    chart.update()
}

async function fetch_update(params) {
  // const params = {
  //   k: 10,
  //   p: 10,
  //   t: 0.6,
  //   s: 0.6,
  // }
  const paramsStr = Object.keys(params).map((k) => `${k}=${params[k]}`).join('&')

  const res = await fetch(`/api/events.json?${paramsStr}`)
  if (!res.ok) {
    console.log(res)
    return
  }

  const data = await res.json()
  const events_impact = data.map((event) => {
    const key = event.term
    const values = event.impact.map((impact) => [+(new Date(impact.date)), impact.value])
    return { key, values }
  })

  console.log(data)
  update(events_impact)
}
