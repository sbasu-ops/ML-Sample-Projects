var comp_desc = d3.csv("comp_desc.csv", function(d) { 
    return{
        security_des:d.Symbols,
    }
});

Promise.all([comp_desc
    // enter code to read files
]).then(function(data){
    d3.select("#selectStock")
    .selectAll('myOptions')
    .data(data[0])
    .enter()
    .append('option')
    .text(function (d) { return d.security_des; })
    .attr("value", function (d) { return d.security_des; })
    .attr("selected",function(d){if (d.security_des==='AAPL'){return "selected"}});

}).catch(function(error) {
    console.log(error);
    });


var width= 1200;
var height = 1000;

var svg = d3.select("#net_viz").append("svg")
        .attr("width", width)
        .attr("height", height);

var width_b = 450;
var height_b= 800;

var svg_b = d3.select("#bar_viz").append("svg")
        .attr("width", width_b)
        .attr("height", height_b);


function create_bar(stock_name,nebr_name,ann_retr){
    padding = 20;
    factor = 0.7;
    move = 100

    //creating scales for bar chart
    var ybar=d3.scaleLinear().range([height_b*factor-2*padding,padding]);
    var xbar=d3.scaleBand().range([2*padding,width_b-1*padding]);
    
    nebr_name.push(stock_name)
    xbar.domain(nebr_name.map(function(d){return d;}));
    
    if (d3.max(ann_retr, function(d) {return d; })<=0){
           console.log('key')
           ybar.domain([d3.min(ann_retr, function(d) {return d; }),0]);
    }
    else if (d3.min(ann_retr, function(d) {return d; })>0) {
           console.log('allpositive')
           ybar.domain([0,d3.max(ann_retr, function(d) {return d; })]);
    } else {
        console.log('notkey')
        ybar.domain([d3.min(ann_retr, function(d) {return d; }),d3.max(ann_retr, function(d) {return d; })]);
    } 
    
    y_zero=ybar(0);

    //creating y-axis
    
     svg_b.append("g")
    .attr("id","y-axis-bars")
    .attr("class","y-axis-bars")
    .attr("transform", "translate(" + String(2*padding) + "," +String(move+0*padding)+")")
    .call(d3.axisLeft(ybar).tickFormat(d3.format("d")));
    


    // Add the text label for Y Axis
    svg_b.append("text")
    .attr("class", "bar_y_axis_label")
    .attr("id", "bar_y_axis_label")
    .attr("text-anchor", "start")
    .attr("transform", "rotate(-90)")
    .attr("x", -y_zero-move-2*padding)
    .attr("y", 0.5*padding)
    .text("Annual Returns (%)")
    .style('fill',"black")
    .style('font-size',12)
    

    //creating x-axis

    svg_b.append("g")
    .attr("id","x-axis-bars")
    .attr("class","x-axis-bars")
    .attr("transform", "translate(0," + String(move+y_zero) + ")")
    .call(d3.axisBottom(xbar));
   

    // Add the text label for X axis
    svg_b.append("text")
    .attr("text-anchor", "center")
    .attr("x", (width_b-3*padding)/2)
    .attr("y", move+y_zero + 2*padding)
    .text("Stock Ticker")
    .attr("class", "bar_x_axis_label")
    .attr("id", "bar_x_axis_label")
    .style('fill',"black")
    .style('font-size',12)
    
    
    //formatting the tick
    d3.selectAll(".tick>text")
        .each(function(d, i){
             d3.select(this).style("font-size","12px");
             d3.select(this).style("fill","black");
            
   });

    //adding the bars

    svg_b.append('g')
    .attr('id','bars')
    .selectAll("rect")
    .data(ann_retr)
    .enter()
    .append("rect")
    .attr("fill", function(d){ 
        if (d>=0){ return "green";}
        else {return 'red';}
    })
    .attr("fill-opacity",0.7)
    .attr('x',function(d,i){return xbar(nebr_name[i])+22})
    .attr('y',function(d){
        
        if (d>=0){return ybar(d)+move;}
        else {return y_zero+move;}
    })
    .attr('height', function(d){
        if (d>=0){return y_zero-ybar(d)}
        else {return ybar(d)-y_zero}
        })
    .attr('width',20);

    //adding chart title

    svg_b.append("text")
     .attr("class", "chart_title")
     .attr("id",'chart_title')
     .attr("text-anchor", "center")
     .attr("x", (width_b-10*padding)/2)
     .attr("y", 5*padding)
     .text("1-year Performance Comparison")
     .style('fill',"red")
     .style('font-size',16)
     .style("font-weight","bolder");
   
};





function run_func(){
    
    month = document.getElementById("gMonth").options[document.getElementById("gMonth").selectedIndex].text;
    year =  document.getElementById("gYear").options[document.getElementById("gYear").selectedIndex].text;
    stock = document.getElementById("selectStock").options[document.getElementById("selectStock").selectedIndex].text;
    var view_type= 1;
    if (document.getElementById("dense").checked === true){view_type=0};
    
    var data_path = "../nearest_neighbor/" + String(month)+'_'+String(year)+'.csv'
    var info_path = "../nearest_neighbor/" + String(month)+'_'+String(year)+'_info.csv'
    
   
    
    var main_data1 = d3.csv(data_path,function(d){
        return{
            source:+d['Source'],
            target: +d['Target'],
        }
    });
    
    var info_data1 = d3.csv(info_path,function(d){
        return{
            ind: +d['index'],
            ticker: d.Ticker,
            mkt_cap: +d['Mkt Cap'],
            sector: d.Sector,
            price: +d['Price'],
            pe: +d['P/E'],
            returns: +d['Returns']
        }
    });
    

    svg.selectAll('*').remove();
    svg_b.selectAll('*').remove();

    Promise.all([main_data1,info_data1,stock,view_type
        // enter code to read files
    ]).then(function(data){
        
        create_vis(data[0],data[1],data[2],data[3]);
    }).catch(function(error) {
        console.log(error);
        });

    
    
}


function create_vis(data,info,stk_ticker,view){
    
    
    var links =  data;
    var infors = info;
    var nodes = {}; 
    
    
    

    links.forEach(function(link) {
        link.source = nodes[link.source] || (nodes[link.source] = {name: link.source});
        link.target = nodes[link.target] || (nodes[link.target] = {name: link.target});
    });

    
    
    
    d3.values(infors).forEach(function(infor){
        if (infor.ind !== undefined){
            
            if (nodes[infor.ind]['name']===infor.ind){
                nodes[infor.ind]['mktcap']= infor.mkt_cap;
                nodes[infor.ind]['pe']= infor.pe;
                nodes[infor.ind]['sector']= infor.sector;
                nodes[infor.ind]['ticker']=infor.ticker;
                nodes[infor.ind]['returns']=infor.returns;
           }
        }
    });

    //selecting the visualization sectors
    var sector_names=[];
    var nbr_name=[];
    var nbr_index=[];
    var annual_returns= [];

    links.map(item => {
        if (item['source']['ticker']===stk_ticker){
            console.log(stk_ticker);
            nbr_name.push(item['target']['ticker']);
            nbr_index.push(item['target']['name'])
            annual_returns.push(item['target']['returns'])
            
            if (!annual_returns.includes(item['source']['returns'])){
                annual_returns.push(item['source']['returns'])
            }
            if (!sector_names.includes(item['target']['sector'])){
                sector_names.push(item['target']['sector'])
            }

            if (!sector_names.includes(item['source']['sector'])){
                sector_names.push(item['source']['sector'])
                
            }  
    }});
    
    console.log(nbr_name);
    console.log(nbr_index);
    console.log(annual_returns);
    dummy_nbr_name=nbr_name

    if ((sector_names.length===0) | (view===0)){
        var sector_names=['Communication Services','Consumer Discretionary','Consumer Staples','Energy','Financials','Health Care','Industrials','Information Technology','Materials','Real Estate','Utilities'];
    }
    
  
    //defining the scale for radius with range and domain using Market Cap
    var rad = d3.scaleLog().range([2,12]);
    rad.domain([d3.min(d3.values(nodes), function(d) { return d.mktcap; }),d3.max(d3.values(nodes), function(d) { return d.mktcap; })]);
    
    //creating different node/circle classes based on P/E ratio
    
    d3.values(nodes).forEach(function(node){
        if (node.sector==='Communication Sevices'){
            node.type='commsr';
        } else if (node.sector==='Consumer Discretionary'){
            node.type='consdis';
        } else if (node.sector==='Consumer Staples'){
            node.type='consstap';
        } else if (node.sector==='Energy'){
            node.type='energy';
        } else if (node.sector==='Financials'){
            node.type='financials';
        } else if (node.sector==='Health Care'){
            node.type='health';
        } else if (node.sector==='Industrials'){
            node.type='industrials';
        } else if (node.sector==='Information Technology'){
            node.type='infotech';
        } else if (node.sector==='Materials'){
            node.type='materials';
        } else if (node.sector==='Real Estate'){
            node.type='realestate';
        } else if (node.sector==='Utilities'){
            node.type='utilities';
        } else {
            node.type='missing';
        }

       if (node.ticker===stk_ticker){
            node.type='selected';
        }
        
        
       if (nbr_name.includes(node.ticker)){
            node.type='neighbors';
        }
    });

    if (stk_ticker !== 'all'){
        
        create_bar(stk_ticker,dummy_nbr_name,annual_returns);
    }
    
    //force layout
    var force = d3.forceSimulation()
    .nodes(d3.values(nodes))
    .force("link", d3.forceLink(links).distance(2))
    .force('center', d3.forceCenter((width /2)-60, (height / 2)+50))
    .force("x", d3.forceX())
    .force("y", d3.forceY())
    .force("charge", d3.forceManyBody().strength(-7))
    .alphaTarget(1)
    .on("tick", tick);
    
    
    //define the edges
    var path = svg.append("g")
    .selectAll("path")
    .data(links)
    .enter()
    .append("path")
    .attr("class", "edges")
    .attr("id","edges")
    .attr("opacity",function(d){
        if(sector_names.includes(d['source']['sector']) & sector_names.includes(d['target']['sector'])){return 1}
        else {return 0}
    });
    
   

    // define the nodes
    var node = svg.selectAll(".node")
    .data(force.nodes())
    .enter().append("g")
    .attr("class", "node")
    .attr("id","node")
    .on('dblclick',function(d){
        if (d.newtype==='pinned'){
          d.newtype=null;
          d.fx=null;
          d.fy=null;
          d.fixed=false;
          d3.select(this).select("circle").classed(d.type,true);
          d3.select(this).select("circle").classed("pinclass",false);
          
        }
     })
    .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // add the nodes
    node.append("circle")
    .attr("id", "cir_node")
    .attr("opacity",function(d){
        if (sector_names.includes(d.sector)){return 1}
        else {return 0}
    })
    .attr("r", function(d){
        return rad(d.mktcap);
     })
    .attr("class", function(d) {
       return  d.type; 
    });

    //adding node labels
    var label = svg.selectAll("mytext")
    .data(force.nodes())
    .enter()
    .append("text")
    .attr("opacity",function(d){
        if (sector_names.includes(d.sector)){return 1}
        else {return 0}})
    .text(function (d) { return d.ticker; })
    .attr("class","node_label");

    //adding legend
    
    // create a list of keys
    var keys = ['Communication Services','Consumer Discretionary','Consumer Staples','Energy','Financials','Health Care','Industrials','Information Technology','Materials','Real Estate','Utilities','N/A','Neighbors','Selected Ticker'].reverse();

    // Usually you have a color scale in your chart already
    var color = ["#0000ff","#000000",'#d6d2d2','#a50026','#d73027','#f46d43','#fdae61','#d3af4a','#696904','#c5f710','#a6d96a','#66bd63','#1a9850','#006837']

    // Add one dot in the legend for each name.
    svg.selectAll("mydots")
    .data(keys)
    .enter()
    .append("circle")
        .attr("cx", 980)
        .attr("cy", function(d,i){ return 150 + i*25}) 
        .attr("r", 7)
        .style("fill", function(d,i){ return color[i]})

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
    .data(keys)
    .enter()
    .append("text")
        .attr("x", 1000)
        .attr("y", function(d,i){ return 150 + i*25})
        .attr("class", "legend")
        .attr("id",'legend') 
        .style("fill", function(d,i){ return color[i]})
        .text(function(d){ return d})
        .attr("text-anchor", "left")
        .style("alignment-baseline", "middle")

     //creating sizing information note
     svg.append("text")
     .attr("class", "size")
     .attr("id",'size')
     .attr("text-anchor", "center")
     .attr("x", 980)
     .attr("y", 550)
     .text("Node sized by market cap");

     // add the curvy lines
    function tick() {
        path.attr("d", function(d) {
            var dx = d.target.x - d.source.x,
                dy = d.target.y - d.source.y,
                dr = Math.sqrt(dx * dx + dy * dy);
            return "M" +
                d.source.x + "," +
                d.source.y + "A" +
                dr + "," + dr + " 0 0,1 " +
                d.target.x + "," +
                d.target.y;
        });

        node.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")"; 
        });

        label.attr("x", function(d){ return d.x + 10; })
    		.attr("y", function (d) {return d.y - 10; });
    };

    function dragstarted(d) {
        if (!d3.event.active) force.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    };

    function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
        d.fixed=true;
    };

    function dragended(d) {
        if (!d3.event.active) force.alphaTarget(0);
        if (d.fixed == true) {
            d.fx = d.x;
            d.fy = d.y;
            d3.select(this).select("circle").classed("pinclass", !d3.select(this).classed("pinclass"));
            d.newtype='pinned';
        }
        else {
            d.fx = null;
            d.fy = null;
        }
    };

    return 
};





var main_data = d3.csv("../nearest_neighbor/January_2016.csv",function(d){
    return{
        source:+d['Source'],
        target: +d['Target'],
    }
});

var info_data = d3.csv("../nearest_neighbor/January_2016_info.csv",function(d){
    return{
        ind: +d['index'],
        ticker: d.Ticker,
        mkt_cap: +d['Mkt Cap'],
        sector: d.Sector,
        price: +d['Price'],
        pe: +d['P/E'],
        returns: +d['Returns']
    }
});

Promise.all([main_data,info_data
    // enter code to read files
]).then(function(data){
    create_vis(data[0],data[1],'all',0);
}).catch(function(error) {
    console.log(error);
    });





