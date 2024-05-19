import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

doctor_img = get_base64_image("./doctor.png")
you_img = get_base64_image("./you.jpg")

css = f'''
<style>
body {{
    background: linear-gradient(to right, #e0f7fa, #b2ebf2);
    font-family: 'Arial', sans-serif;
}}

.chat-container {{
    max-width: 800px;
    margin: auto;
    padding: 2rem;
    background-color: #f5f5f5;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

.chat-message {{
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}}

.chat-message.user {{
    background-color: #d1e7ff;
}}

.chat-message.bot {{
    background-color: #e0f7fa;
}}

.chat-message .avatar {{
    width: 20%;
    display: flex;
    justify-content: center;
    align-items: center;
}}

.chat-message .avatar img {{
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}}

.chat-message .message {{
    width: 80%;
    padding: 0 1.5rem;
    color: #333;
}}

.sidebar .old-tests {{
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}}

.sidebar .old-tests h4 {{
    margin-top: 0;
    color: #333;
}}

.sidebar .old-tests p {{
    margin: 0;
    color: #666;
}}
</style>
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{doctor_img}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,{you_img}">
    </div>    
    <div class="message">{{{{MSG}}}}</div>
</div>
'''

