/* global d3, nv, event_impact */

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

  pattern
    .append('rect')
    .attr('height', '100%')
    .attr('width', '100%')
    .attr('fill', bg)

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

  // add one polygon for each data in event_impact
  svg
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .attr('viewBox', `0 0 ${config.width} ${config.height}`)
    .style('width', '100%')
    .style('height', CSS.percent(100 * config.height / config.width))
    .style('background-color', '#333')
    .selectAll('polygon')
    .data(event_impact)
    .enter()
    .append('polygon')
    .attr('fill', (d) => `url("#pattern-noangle--${slugifyToId(d.key)}")`)
    .attr('stroke', (d) => d.color)
    .attr('points', (singleEventImpact) => {
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
    })
}


event_impact.sort((a, b) => {
  // sort by max .values
  const maxA = d3.max(a.values, d => d[1])
  const maxB = d3.max(b.values, d => d[1])
  return maxA - maxB
})


const colors = d3.scale.category20()

var chart
nv.addGraph(function() {
  chart = nv.models.stackedAreaChart()

  // const _update = chart.update
  // function onUpdate() {
  //     console.log('update');
  //     _update.apply(chart, arguments)
  // }
  // chart.update = onUpdate

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

  const svg = d3.select('#chart1')

  const defs = svg.append('defs')

  d3.select('#chart1')
    .datum(event_impact.map((d) => {
      d.classed = 'lorem'
      return d
    }))
    .each(console.log)
    .call(chart)

  nv.utils.windowResize(chart.update)

  d3.selectAll('#chart1 .nv-area').each((x) => console.log(x))

  const nvAreas = d3.selectAll('#chart1 .nv-area')
  nvAreas.attr('custom-fill', (d) => {
    return slugifyToId(d.key)
  })
  const styleElement = d3.select('#chart1').append('style')
  let css = ''
  nvAreas.each((nvArea) => {
    console.log(nvArea)
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
    textPatternFactory(defs, `pattern-hover--${id}`, {
      text, bg, angle,
      fg: fgHover,
    })
    textPatternFactory(defs, `pattern-noangle--${id}`, {
      text, bg,
      fg: bg.l > 0.5 ? '#000' : '#fff',
      angle: 0,
      lineHeight: 1.1,
    })
    // css += `.nv-area-${index} { fill: url("#pattern--${id}") !important; }` + '\n'
    // css += `.nv-area-${index}:hover { fill: url("#pattern-hover--${id}") !important; }` + '\n'
    css += `[custom-fill="${id}"] { fill: url("#pattern--${id}") !important; }` + '\n'
    css += `[custom-fill="${id}"]:hover { fill: url("#pattern-hover--${id}") !important; }` + '\n'
  })
  styleElement.text(css)
  // const nvAreas = d3.selectAll('#chart1 .nv-area')
  //     .style('fill', (d, index) => `url("#pattern-${index}")`)
  // nvAreas.on("mouseover", function () {
  //     d3.select(this).raise();
  // });
  // nvAreas.on("mouseout", function () {
  //     d3.select(this).lower();
  // });

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

  chart.update()
  return chart
})
