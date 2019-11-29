import IPython
from google.colab import output

chatbot_html = """
<style type="text/css">#log p { margin: 5px; font-family: sans-serif; }</style>
<div id="log"
     style="box-sizing: border-box;
            width: 600px;
            height: 32em;
            border: 1px grey solid;
            padding: 2px;
            overflow: scroll;">
</div>
<input type="text" id="typehere" placeholder="Введите текст!"
       style="box-sizing: border-box;
              width: 600px;
              margin-top: 5px;">
<script>
function paraWithText(t) {
    let tn = document.createTextNode(t);
    let ptag = document.createElement('p');
    ptag.appendChild(tn);
    return ptag;
}
document.querySelector('#typehere').onchange = async function() {
    let inputField = document.querySelector('#typehere');
    let val = inputField.value;
    inputField.value = "";
    let resp = await getResp(val);
    let objDiv = document.getElementById("log");
    objDiv.appendChild(paraWithText('You: ' + val));
    objDiv.appendChild(paraWithText('Textbot: ' + resp));
    objDiv.scrollTop = objDiv.scrollHeight;
};
async function colabGetResp(val) {
    let resp = await google.colab.kernel.invokeFunction(
        'notebook.get_response', [val], {});
    return resp.data['application/json']['result'];
}
async function webGetResp(val) {
    let resp = await fetch("/response.json?sentence=" + 
        encodeURIComponent(val));
    let data = await resp.json();
    return data['result'];
}
</script>
"""

def Display(GenerateMethod, val=None):
  '''Метод для запуска чатбота.
  
  ::param::GenerateMethod - любой метод, принимающий на вход последнее высказывание диалога, а возвращающее свое высказывание в str.
  
  '''
  display(IPython.display.HTML(chatbot_html + \
                             "<script>let getResp = colabGetResp;</script>"))

  def get_response(val):
    resp = GenerateMethod(val)
    return IPython.display.JSON({'result': resp})

  output.register_callback('notebook.get_response', get_response)
