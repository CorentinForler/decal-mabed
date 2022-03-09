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
function textPatternFactory(defs, id, { text, fg, bg, strokeColor = 'none', angle = 0, lineHeight = 1.6, fontSize = 15 }) {
  const textAttrs = {
    'font-size': fontSize,
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

    // chart.xAxis.tickFormat(function(d) { return d3.time.format('%x')(new Date(d)) })
    chart.xAxis.tickFormat(function(d) { return d3.time.format('%d/%m/%Y')(new Date(d)) })

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
  if (event_table) {
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
  }

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

  const form = document.querySelector('form#interactive-form')
  form.addEventListener('submit', async function (e) {
    // prevent submit
    e.preventDefault()

    const values = {}
    for (const input of form.querySelectorAll('input[name],select[name]')) {
      values[input.name] = input.value
    }

    if (values.tsl__unit) {
      const factor = {
        minutes: 1,
        hours: 60,
        days: 60 * 24,
      }
      values.tsl = values.tsl * factor[values.tsl__unit]
      values.tsl__unit = 'minutes'
    }


    const spinner = form.querySelector('.spinner-container')
    const errorContainer = form.querySelector('.error-container')

    errorContainer.setAttribute('hidden', 'hidden')
    errorContainer.innerHTML = ''
    spinner.removeAttribute('hidden')

    try {
      await fetch_update(values)
    } catch(e) {
      errorContainer.innerHTML = `${e.message}<br />${e.stack}`
      errorContainer.removeAttribute('hidden')
    } finally {
      spinner.setAttribute('hidden', 'hidden')
    }
  })
}

function update(event_impact = [], raw_events = null, opts = {}) {
  const rainbow = Array.from(
    { length: event_impact.length },
    (_, i) => {
      const h = (i / event_impact.length) * 360
      const s = 1
      const lab = d3.lab(d3.hsl(h, s, 0.5))
      lab.l = 50
      return lab.toString()
    }
  )
  const colors = (i) => rainbow[i]
  // const colors = d3.scale.linear()
  //   .domain([0, event_impact.length])
  //   .range([d3.hsl(0, 1, 0.5), d3.hsl(360, 1, 0.5)])

  const svg = d3.select('#chart1')

  let defs = svg.select('defs.all-defs')
  if (!defs.node()) {
    defs = svg.append('defs').attr('class', 'all-defs')
  } else {
    defs.html('') // empty
  }

  if (raw_events) {
    const eventChartId = row => `table-impact--event-${slugifyToId(row.term + '_' + row.mag)}`

    const abbr = (text, hoverText, cl='') => `<abbr class="${cl}" title="${hoverText.replace('"','\\"')}">${text}</abbr>`
    tabulate(raw_events, ['Event Terms', 'Proposed articles', 'Event impact'], {
      tableElement: d3.select('#event_table'),
      css: `
        #event_table {
          border-collapse: collapse;
          border: 1px solid #ddd;
          font-size: 18px;
          line-height: 1.4;
          position: sticky;
        }
        #event_table td, th {
          border: 2px solid black;
          _border: 1px solid #ddd;
          padding: 4px;
        }
        #event_table thead {
          background-color: #eee;
          position: sticky;
          top: 0px;
          z-index: 1;
          box-shadow: 0 2px 0 0 #000, 0 -2px 0 0 #000;
          border: 0px solid black !important;
        }
        #event_table tr:nth-child(even) {
          background-color: hsla(0, 0%, 50%, 0.1);
        }
        #event_table th:first-child {
          width: 180px;
        }
        #event_table tbody tr {
          border: 2px solid black;
        }
        #event_table tbody tr td {
          padding: 0 2px;
        }
        #event_table .term, #event_table .related-term {
          white-space: nowrap;
        }
        #event_table .term {
          /*color: #ff7273;*/
          color: #004bbd;
          font-weight: 900;
          /*text-decoration: underline;*/
          /*text-decoration-style: dotted;*/
        }
        #event_table .related-term {
          /*color: #00adff;*/
          color: #8a4000;
          font-weight: 600;
          font-style: italic;
          text-decoration: none;
        }


        #event_table .event-proposed-article-wrapper input {
          width: 1.6em;
          height: 1.6em;
          padding: 0;
          margin: 0;
        }
        #event_table .event-proposed-article-wrapper button {
          padding: 0;
          margin: 0;
          line-height: 0px;
          font-family: Inter;
          font-weight: 900;
          font-size: 14px;
        }
        #event_table .event-proposed-article-wrapper > div.controls {
          display: flex;
          flex-direction: column;
          justify-content: center;
          width: auto;
          height: 100%;
          margin-top: 0.2em;
          margin-right: 0.4em;
        }
        #event_table .event-proposed-article-wrapper {
          display: flex;
          flex-direction: row;
          align-items: flex-start;
          justify-content: flex-start;

          padding: 1em 2px;

          _padding-left: 1.5em;
          _margin-left: -1.5em;
          _margin-right: -2px;
          _padding-right: 2px;
          text-align: left;
        }
        #event_table .event-proposed-article-wrapper[data-correct="true"] input[type="checkbox"] {
          accent-color: #326d45;
        }
        #event_table .event-proposed-article-wrapper[data-correct="true"] /*.event-proposed-article*/ {
          background-color: hsla(140, 100%, 50%, 0.1);
          border-radius: 5px;
        }
        #event_table .event-proposed-article-wrapper[data-correct="false"] /*.event-proposed-article*/ {
          text-decoration: line-through !important;
          text-decoration-color: rgba(0, 0, 0, 0.5);
          opacity: 0.8;
        }
      `,
      mapper(v, column, row) {
        if (column === 'Event Terms') {
          const term = row.term
          const related = row.related

          let html = ''

          const f = (term) => `<span class="term">${term}</span>`
          html += term.split(', ').map(f).join(', ')

          if (Array.isArray(related)) {
            const f = ({ term, mag }) => abbr(term, mag.toString(), 'related-term')
            html += ", " + related.map(f).join(', ')
            // return v.map(({ term, mag }) => `${term}(${mag})`).join(', ')
          }

          return { html }
        }
        if (column === 'start' || column === 'end') {
          const splits = v.split(' ')
          if (splits[1] === '00:00:00') return splits[0]
          return splits.join('\n')
        }
        if (column === 'Event impact') {
          return { html: `<div style="width: 450px; height: 100%; aspect-ratio: 16/9;" id="${eventChartId(row)}"></div>` }
        }
        if (column === 'Proposed articles') {
          v = row.articles
          if (!v) return ''
          if (!Array.isArray(v)) {
            v = [v]
          }

          const regexMain = new RegExp(`(?<!")\\b(${
            row.term.split(', ').join('|')
          })\\b(?!")`, 'ig')

          const regexRelated = new RegExp(`(?<!")\\b(${
            row.related.map(({ term }) => term).join('|')
          })\\b(?!")`, 'ig')

          const uniq = (x, i, arr) => arr.indexOf(x) === i

          const replaces = [{
            regex: regexMain,
            replace: '<span class="term">$1</span>',
          }, {
            regex: regexRelated,
            replace: '<span class="related-term">$1</span>',
          }, {
            regex: /(\(ap\) \p{Pd} )/igu,
            replace: '',
          }]

          const replace = (s) => replaces.reduce((acc, r) => acc.replace(r.regex, r.replace), s)

          const checkboxWrap = (html, i) => `<label class="event-proposed-article-wrapper">
            <div class="controls">
              <input type="checkbox" data-indeterminate="true" />
              <button>↑</button>
              <button>↓</button>
            </div>
            <div class="event-proposed-article">${html}</div>
          </label>`

          const html = v
            .map(t => {
              const [score, x] = Array.isArray(t) ? t : ["?", t]
              // const countMainTerms = (x.match(regexMain) || []).map(y => y.toLowerCase()).filter(uniq).length
              // const countRelatedTerms = (x.match(regexRelated) || []).map(y => y.toLowerCase()).filter(uniq).length
              // return `score=${score}, #main=${countMainTerms}, #related=${countRelatedTerms}<br/>${replace(x)}`
              return replace(x)
            })
            .map(checkboxWrap)
            .join('')
            // .join('<br/>')

          return { html }
          // return v
          // return { html: abbr(v.substr(0, 70) + "…", v) }
        }
        return v
      }
    })

    document.querySelectorAll('input[type="checkbox"][data-indeterminate="true"]').forEach(input => {
      input.removeAttribute('data-indeterminate')
      input.indeterminate = true
      input.checked = false
      input.addEventListener('change', () => {
        input.parentElement.parentElement.setAttribute('data-correct', input.checked ? 'true' : 'false')
      })
    })

    const colProposedArticles = document.querySelector('#event_table thead th:nth-child(2)')
    colProposedArticles.style.position = 'relative'
    colProposedArticles.innerHTML += `<div style="position:absolute;top:0px;left:4px;line-height:1.85em;opacity:0.8;"><input type=checkbox disabled checked/>Correct?</div>`

    let maxImpact = 0
    for (const event of raw_events) {
      for (const { value } of event.impact) {
        if (value > maxImpact) {
          maxImpact = value
        }
      }
    }
    const k = Math.pow(10, Math.floor(Math.log10(maxImpact)))
    maxImpact = Math.ceil(maxImpact / k) * k

    let eventIndex = -1
    for (const event of raw_events) {
      eventIndex += 1
      const dates = []
      const values = []

      const imp = event.impact.slice() // shallow copy
      // while (imp.length > 0 && imp[0].value == 0) {
      //   imp.shift()
      // }
      // while (imp.length > 0 && imp[imp.length - 1].value == 0) {
      //   imp.pop()
      // }

      for (const { date, value } of imp) {
        dates.push(new Date(date.replace(' ', 'T') + 'Z'))
        values.push(Math.round(value * 100) / 100)
      }

      // https://naver.github.io/billboard.js/release/latest/doc/Options.html
      // generate the chart
      bb.generate({
        bindto: "#" + eventChartId(event),
        data: {
          // type: "area-spline",
          type: "area",
          x: "x",
          columns: [
            ["x", ...dates],
            ["impact", ...values],
          ],
          colors: { impact: colors(eventIndex) },
        },
        axis: {
          x: {
            type: "timeseries",
            min: dates[0],
            max: dates[dates.length - 1],
            padding: { left: 0, right: 0 },
            tick: {
              format: "%Y-%m-%d",
              fit: true,
              count: 8,
            },
            clipPath: false,
          },
          y: {
            label: 'Impact',
            max: maxImpact,
            min: 0,
            padding: { top: 0, bottom: 0 },
            clipPath: false,
          },
        },
        zoom: {
          enabled: true
        },
        legend: {
          show: false,
        },
        point: {
          show: false,
        }
      });
    }
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
    const text = nvArea.event.term
    // const text = (nvArea.articles && nvArea.articles[0]) || nvArea.key

    // const id = index
    const id = slugifyToId(nvArea.key)

    nvArea.color = d3.lab(colors(index)).darker(1).toString()

    let bg
    if (nvArea.color.indexOf('#') === 0) {
      // if nvArea.color is a color
      bg = d3.lab(nvArea.color)
    } else {
      bg = d3.lab(colors(index))
    }
    const angle = -45 * bg.h / 360
    // const fg = bg.l > 50 ? bg.darker(4) : bg.brighter(2)
    const fg = bg.l > 50 ? "#000" : "#fff";
    const fgHover = bg.l > 50 ? bg.darker(1) : bg.brighter(1)

    textPatternFactory(defs, `pattern--${id}`, {
      text, fg, fgHover, bg, angle, fontSize: 15, lineHeight: 1.5,
    })
    // textPatternFactory(defs, `pattern-hover--${id}`, {
    //   text, bg, angle,
    //   fg: fgHover,
    // })
    // textPatternFactory(defs, `pattern-noangle--${id}`, {
    //   text, bg,
    //   fg: bg.l > 50 ? '#000' : '#fff',
    //   angle: 0,
    //   lineHeight: 1.1,
    // })
    textPatternFactory(defs, `pattern-textonly--${id}`, {
      text, bg: 'transparent',
      fg: bg.l > 50 ? '#000' : '#fff',
      // strokeColor: bg.l < 50 ? '#000' : '#fff',
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

  if (opts.zoom) {
    chart.xDomain(opts.zoom).update()
  }

  chart.update()
}

function tabulate(data, columns, opts = {}) {
  const randomId = Math.random().toString(36).substring(2)
  if (typeof opts.mapper !== 'function') {
    opts.mapper = (x) => x
  }
  if (!opts.tableElement) {
    opts.tableElement = null
  }
  if (!opts.css) {
    opts.css = `
      table.tabulated-${randomId} {
        border-collapse: collapse;
        border: 1px solid #ddd;
        font-family: sans-serif;
        font-size: 12px;
      }
      table.tabulated-${randomId} td, th {
        border: 1px solid #ddd;
        padding: 5px;
      }
      table.tabulated-${randomId} thead {
        background-color: #eee;
      }
      table.tabulated-${randomId} tr:nth-child(even) {
        background-color: hsla(220, 100%, 50%, 0.01);
      }
    `
  }

  // https://stackoverflow.com/a/18072266
	const table = opts.tableElement ? opts.tableElement : d3.select('body').append('table')
  table.attr('class', `tabulated-${randomId}`)

  table.selectAll('*').remove()

	const thead = table.append('thead')
	const	tbody = table.append('tbody')

  // append the css
  table.append('style')
    .text(opts.css)

	// append the header row
	thead.append('tr')
	  .selectAll('th')
	  .data(columns).enter()
	  .append('th')
	    .text(column => column)

	// create a row for each object in the data
	const rows = tbody.selectAll('tr')
	  .data(data)
	  .enter()
	  .append('tr')

	// create a cell in each row for each column
	const cells = rows.selectAll('td')
	  .data(row => columns.map(
      column => ({
        column,
        value: opts.mapper(row[column], column, row),
	    })
    ))
	  .enter()
	  .append('td')
	    .html(d => {
        if (typeof d.value === 'object' && typeof d.value.html === 'string') {
          return d.value.html || ''
        } else {
          return (d.value || '').toString()
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('&', '&amp;')
        }
      })

  return table
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

  let data = await res.json()

  // data = [data[0], data[1], data[3]]
  const raw_events = data

  window._data = data

  const event_impact = data.map((event) => {
    const key = event.term
    const values = event.impact
      .map((impact) => {
        const k = +(new Date(impact.date))
        const v = Math.round(impact.value * 100) / 100
        return [k, v]
      })
    if (Array.isArray(event.articles)) {
      if (Array.isArray(event.articles[0])) {
        // score, article
        const articles = event.articles.map(x => x[1])
        return { key: articles[0], values, articles, event }
      } else {
        return { key: event.articles[0], values, articles: event.articles, event }
      }
    } else if (event.articles) {
      let desc = event.articles
      const i = desc.toLowerCase().indexOf("(ap)")
      if (i >= 0) { desc = desc.substring(i + 7).trim() }
      return { key, values, articles: [desc], event }
    } else {
      return { key, values, articles: [], event }
    }
  })

  // Auto-zoom
  let min_date = +Infinity
  let max_date = -Infinity
  for (const event of data) {
    for (const impact of event.impact) {
      if (impact.value > 0) {
        const date = new Date(impact.date)
        if (date < min_date) {
          min_date = date
        }
        if (date > max_date) {
          max_date = date
        }
      }
    }

    // const start = new Date(event.start)
    // const end = new Date(event.end)
    // if (start < min_date) { min_date = start }
    // if (end > max_date) { max_date = end }
  }

  let zoom = null
  if (min_date <= max_date) {
    // min_date = new Date((+min_date) - 24 * 3600 * 1000)
    // max_date = new Date((+max_date) + 24 * 3600 * 1000)
    zoom = [min_date, max_date]
  }

  // console.log({zoom})

  update(event_impact, raw_events, { zoom })
}
