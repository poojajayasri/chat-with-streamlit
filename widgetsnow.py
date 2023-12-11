import streamlit as st
import json
import os
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit_option_menu
from streamlit_cookies_manager import EncryptedCookieManager
#from streamlit_login_authtime_ui.streamlitcookiesmanagermain.streamlit_cookies_manager import EncryptedCookieManager
#from streamlit_login_authtime_ui.cookiesmain.streamlit_cookies_manager import EncryptedCookieManager
from utilsnow import check_usr_pass
from utilsnow import load_lottieurl
from utilsnow import check_valid_name
from utilsnow import check_valid_email
from utilsnow import register_new_usr1

from utilsnow import check_unique_email
from utilsnow import check_unique_usr
from utilsnow import register_new_usr
from utilsnow import check_email_exists
from utilsnow import generate_random_passwd
from utilsnow import send_passwd_in_email
from utilsnow import change_passwd
from utilsnow import check_current_passwd
from oAuthMain import googleauthenticate
from oAuthMain import googlevalidate

class __login__:
    try:
        """
        Builds the UI for the Login/ Sign Up page.
        """

        def __init__(self, auth_token: str, company_name: str, width, height, logout_button_name: str = 'Logout', hide_menu_bool: bool = False, hide_footer_bool: bool = True ):
            """
            Arguments:
            -----------
            1. self
            2. auth_token : The unique authorization token received from - https://www.courier.com/email-api/
            3. company_name : This is the name of the person/ organization which will send the password reset email.
            4. width : Width of the animation on the login page.
            5. height : Height of the animation on the login page.
            6. logout_button_name : The logout button name.
            7. hide_menu_bool : Pass True if the streamlit menu should be hidden.
            8. hide_footer_bool : Pass True if the 'made with streamlit' footer should be hidden.
            9. lottie_url : The lottie animation you would like to use on the login page. Explore animations at - https://lottiefiles.com/featured
            """
            self.auth_token = auth_token
            self.company_name = company_name
            self.width = width
            self.height = height
            self.logout_button_name = logout_button_name
            self.hide_menu_bool = hide_menu_bool
            self.hide_footer_bool = hide_footer_bool
            #self.lottie_url = lottie_url
            self.cookies = EncryptedCookieManager(
            prefix="streamlit_login_ui_yumminess_cookies",
            password='9d68d6f2-4258-45c9-96eb-2d6bc74ddbb5-d8f49cab-edbb-404a-94d0-b25b1d4a564b12345')

            if not self.cookies.ready():
                st.stop()   


        def check_auth_json_file_exists(self, auth_filename: str) -> bool:
            """
            Checks if the auth file (where the user info is stored) already exists.
            """
            file_names = []
            for path in os.listdir('./'):
                if os.path.isfile(os.path.join('./', path)):
                    file_names.append(path)

            present_files = []
            for file_name in file_names:
                if auth_filename in file_name:
                    present_files.append(file_name)
                        
                present_files = sorted(present_files)
                if len(present_files) > 0:
                    return True
            return False

        def get_username(self):
            if st.session_state['LOGOUT_BUTTON_HIT'] == False:
                fetched_cookies = self.cookies
                if '__streamlit_login_signup_ui_username__' in fetched_cookies.keys():
                    username=fetched_cookies['__streamlit_login_signup_ui_username__']
                    return username
    

        def login_widget(self) -> None:
            """
            Creates the login widget, checks and sets cookies, authenticates the users.
            """
            
            
            # Checks if cookie exists.
            if st.session_state['LOGGED_IN'] == False:
                if st.session_state['LOGOUT_BUTTON_HIT'] == False:
                    fetched_cookies = self.cookies
                    if '__streamlit_login_signup_ui_username__' in fetched_cookies.keys():
                        if fetched_cookies['__streamlit_login_signup_ui_username__'] != '1c9a923f-fb21-4a91-b3f3-5f18e3f0118212345':
                            st.session_state['LOGGED_IN'] = True

            if st.session_state['LOGGED_IN'] == False:
                st.session_state['LOGOUT_BUTTON_HIT'] = False 

                del_login = st.empty()
                with del_login.container():
                    colll1, colll2, colll3 = st.columns(3)
                    

                    with colll1:
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                       
                    st.divider()
                    st.write("")
                    st.write("")

                    st.title('Create your account')
                    col11, col21 = st.columns(2)
                    
                    with st.expander("Register account"):
                        self.sign_up_widget()
                    st.write("")

                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.divider()
                    
                    with st.expander("Already have an account? Log in"):
                        #st.header('Login Form')

                        username = st.text_input("", placeholder = 'Email', key="sdrt")
                        password = st.text_input("", placeholder = 'Password', type = 'password',key="fgxf")

                        st.markdown("###")
                        
                        

                        login_submit_button = st.button(label = 'Login', key="rurdf")
                        

                        if login_submit_button == True:
                            try:
                                authenticate_user_check, user = check_usr_pass(username, password)

                                print(authenticate_user_check)
                                print(user)
                                if authenticate_user_check == False or authenticate_user_check == None:
                                    st.error("Invalid Username or Password!")

                                else:
                                    if user.email_verified:
                                        print("VERIFIED")
                                        st.session_state['LOGGED_IN'] = True
                                        self.cookies['__streamlit_login_signup_ui_username__'] = username
                                        self.cookies.save()
                                        del_login.empty()
                                        st.experimental_rerun()
                                    else:
                                        st.error("Verify email")
                            except Exception as e:
                                #st.write(e)
                                st.error("Incorrect ID/Password")
                    with st.expander("Reset Password"):            
                        self.forgot_password()        
        def logingoogle_widget(self) -> None:
            """
            Creates the login widget, checks and sets cookies, authenticates the users.
            """

            # Checks if cookie exists.
            if st.session_state['LOGGED_IN'] == False:
                if st.session_state['LOGOUT_BUTTON_HIT'] == False:
                    fetched_cookies = self.cookies
                    if '__streamlit_login_signup_ui_username__' in fetched_cookies.keys():
                        if fetched_cookies['__streamlit_login_signup_ui_username__'] != '1c9a923f-fb21-4a91-b3f3-5f18e3f0118212345':
                            st.session_state['LOGGED_IN'] = True

            if st.session_state['LOGGED_IN'] == False:
                st.session_state['LOGOUT_BUTTON_HIT'] = False 
                try:
                    del_login = st.empty()
                    #del_login.title("Welcome")
                    with del_login.container():
                        st.title("Sign in with Google")

                        st.write("")
                        st.write("")

                        #auth_url = str(googleauthenticate())
                        #st.form_submit_button(on_click=redirect_button(auth_url))
                        #login_submit_button1 = st.form_submit_button(label = 'Sign in with Google', type= "secondary")

                        #if login_submit_button1 == True:
                        

                        #auth_url = str(googleauthenticate())
                        #print(f"HIEEEEauthurl{auth_url}")
                        #login = st.form_submit_button("login", on_click=open_support_ticket(auth_url))
                        def redirect_button(url: str, text: str= None, color="white"):
                            st.markdown(
                            f"""
                            <a href="{url}" target="_self">
                                <div style="
                                    display: inline-block;
                                    padding: 0.5em 1em;
                                    width: 15   ;
                                    border: 1px solid #d6d6d6;
                                    border-radius: 30px;
                                    box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.2);
                                    text-decoration: none;">
                                    <img src = "https://cdn.discordapp.com/attachments/852337726904598574/1120926073362862080/googlefinal.png" width=32 height=32/>
                                    {text}
                                </div>
                            </a>
                            """,
                            unsafe_allow_html=True
                            )
                        redirect_button(googleauthenticate(),"  Continue with Google")
                        

                        if redirect_button:
                        
                            #code = st.experimental_get_query_params()['code']
                            #if code:
                            user_email = googlevalidate()
                            print(f"HIEEEE{user_email}")
                            if user_email == False or user_email == None:
                                st.error("Invalid ID")
                            else:
                                print("VERIFIED")
                                try:
                                    register_new_usr1(user_email)
                                except Exception as e:
                                    st.write(" ")
                                st.session_state['LOGGED_IN'] = True
                                user_email = str(user_email)
                                #user_email = user_email.encode('utf-8')


                                self.cookies['__streamlit_login_signup_ui_username__'] = user_email
                                self.cookies.save()
                                del_login.empty()
                                st.experimental_rerun()
                            #else:
                            #    st.write("no code")
                            
                except Exception as e:
                    st.write(" ")
        def logingoogle_widgetone(self) -> None:
            """
            Creates the login widget, checks and sets cookies, authenticates the users.
            """

            # Checks if cookie exists.
            if st.session_state['LOGGED_IN'] == False:
                if st.session_state['LOGOUT_BUTTON_HIT'] == False:
                    fetched_cookies = self.cookies
                    if '__streamlit_login_signup_ui_username__' in fetched_cookies.keys():
                        if fetched_cookies['__streamlit_login_signup_ui_username__'] != '1c9a923f-fb21-4a91-b3f3-5f18e3f0118212345':
                            st.session_state['LOGGED_IN'] = True

            if st.session_state['LOGGED_IN'] == False:
                st.session_state['LOGOUT_BUTTON_HIT'] = False 
                try:
                    del_login = st.empty()
                    #del_login.title("Welcome")
                    with del_login.container():
                        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="font-size: 2.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px; "><span style = "color: white; padding: 5px;"></span><img class="chat-icon" src="https://cdn.discordapp.com/attachments/852337726904598574/1126682090101035019/frilogo11.png" width=54 height=54 style="border-radius: 50px;border: 1px solid #515389; padding: 5px;"></img></span></span></div>', unsafe_allow_html=True)
            
                        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="font-size: 3.3em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 600; justify-content: center; text-align: center; border-radius: 30px; padding: 10px;">Deep Dive into your Data,<span style="color: #cc003d;font-family: Brush Script MT;"> Visually</span>.</span><span style="color: #898989; font-size: 1.5em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;"></span></div>', unsafe_allow_html=True)
            
                        st.markdown('<div style = "justify-content: center; text-align: center;"><span style="color: #a1a0a0; font-size: 1em; font-family: Helvetica Neue; font: 500; font-style: bolder; font-weight: 400; justify-content: center; text-align: center; border-radius: 30px; padding: 5px;"> Generate Interactive Mindmaps and Dynamic Flowcharts with <span style="font-style: bolder;font-weight: 600;">Datamap AI.</span></span></div>', unsafe_allow_html=True)

                        st.write("")
                        st.write("")

                        #auth_url = str(googleauthenticate())
                        #st.form_submit_button(on_click=redirect_button(auth_url))
                        #login_submit_button1 = st.form_submit_button(label = 'Sign in with Google', type= "secondary")

                        #if login_submit_button1 == True:
                        

                        #auth_url = str(googleauthenticate())
                        #print(f"HIEEEEauthurl{auth_url}")
                        #login = st.form_submit_button("login", on_click=open_support_ticket(auth_url))
                        def redirect_button(url: str, text: str= None, color="white"):
                            st.markdown(
                            f"""
                            <a href="{url}" target="_self">
                                <div style = "justify-content: center; text-align: center;"><span style="justify-content: center; text-align: center; border-radius: 30px; "> <span style = "display: inline-block;
                                    padding: 0.5em 1em;
                                    width: 15   ;
                                    border: 1px solid #d6d6d6;
                                    border-radius: 30px;
                                    box-shadow: 0px 0px 5px 0px rgba(0,0,0,0.2);
                                    text-decoration: none; background-color: #403f3f; color: white;"><img src = "https://cdn.discordapp.com/attachments/852337726904598574/1120926073362862080/googlefinal.png" width=32 height=32/>  {text}</span> </span></div>
                                
                            </a>
                            """,
                            unsafe_allow_html=True
                            )
                        c11,c12,c13 = st.columns([2,6,2])
                        with c12:
                            redirect_button(googleauthenticate(),"Get Started")
                        

                        if redirect_button:
                        
                            #code = st.experimental_get_query_params()['code']
                            #if code:
                            user_email = googlevalidate()
                            print(f"HIEEEE{user_email}")
                            if user_email == False or user_email == None:
                                st.error("Invalid ID")
                            else:
                                print("VERIFIED")
                                try:
                                    register_new_usr1(user_email)
                                except Exception as e:
                                    st.write(" ")
                                st.session_state['LOGGED_IN'] = True
                                user_email = str(user_email)
                                #user_email = user_email.encode('utf-8')


                                self.cookies['__streamlit_login_signup_ui_username__'] = user_email
                                self.cookies.save()
                                del_login.empty()
                                st.experimental_rerun()
                            #else:
                            #    st.write("no code")
                            
                except Exception as e:
                    st.write(" ")

        def animation(self) -> None:
            """
            Renders the lottie animation.
            """
            lottie_json = load_lottieurl(self.lottie_url)
            st_lottie(lottie_json, width = self.width, height = self.height)


        def sign_up_widget(self) -> None:
            """
            Creates the sign-up widget and stores the user info in a secure way in the _secret_auth_.json file.
            """
            with st.container():
                
                #st.header('Sign Up Form')
                name_sign_up = st.text_input("", placeholder = 'First Name')
                valid_name_check = check_valid_name(name_sign_up)

                email_sign_up = st.text_input("", placeholder = 'Email')
                valid_email_check = check_valid_email(email_sign_up)
                unique_email_check = check_unique_email(email_sign_up)
                
                username_sign_up = st.text_input("", placeholder = 'Username')
                unique_username_check = check_unique_usr(username_sign_up)

                password_sign_up = st.text_input("", placeholder = 'Password', type = 'password')

                st.markdown("###")
                sign_up_submit_button = st.button(label = 'Register')

                if sign_up_submit_button:
                    if valid_name_check == False:
                        st.error("Please enter a valid name!")

                    elif valid_email_check == False:
                        st.error("Please enter a valid Email!")
                    
                    elif unique_email_check == False:
                        st.error("Email already exists!")
                    
                    elif unique_username_check == False:
                        st.error(f'Username {username_sign_up} already exists!')
                    
                    elif unique_username_check == None:
                        st.error('Please enter a non - empty Username!')

                    if valid_name_check == True:
                        if valid_email_check == True:
                            if unique_email_check == True:
                                if unique_username_check == True:
                                    checkk = register_new_usr(name_sign_up, email_sign_up, username_sign_up, password_sign_up)
                                    if checkk == 1:
                                        st.success("Account created. Please verify your email address")
                                    if checkk == 0:
                                        st.error("Email already exists")


        def forgot_password(self) -> None:
            """
            Creates the forgot password widget and after user authentication (email), triggers an email to the user 
            containing a random password.
            """
            with st.container():
                email_forgot_passwd = st.text_input("", placeholder= 'Email',key= "yoo")
                email_exists_check, username_forgot_passwd = check_email_exists(email_forgot_passwd)

                st.markdown("###")
                forgot_passwd_submit_button = st.button(label = 'Reset')

                if forgot_passwd_submit_button:

                    statuss = change_passwd(email_forgot_passwd)
                    if statuss==200:
                        st.success("Password Reset Link Sent")
                    else:
                        st.error("Please check your email address.")



        def reset_password(self) -> None:
            """
            Creates the reset password widget and after user authentication (email and the password shared over that email), 
            resets the password and updates the same in the _secret_auth_.json file.
            """
            with st.container():
                
                email_reset_passwd = st.text_input("", placeholder= 'Email')
                email_exists_check, username_reset_passwd = check_email_exists(email_reset_passwd)

                current_passwd = st.text_input("Temporary Password", placeholder= 'Please enter the password you received in the email')
                current_passwd_check = check_current_passwd(email_reset_passwd, current_passwd)

                new_passwd = st.text_input("New Password", placeholder= 'Please enter a new, strong password', type = 'password')

                new_passwd_1 = st.text_input("Re - Enter New Password", placeholder= 'Please re- enter the new password', type = 'password')

                st.markdown("###")
                reset_passwd_submit_button = st.button(label = 'Reset Password')

                if reset_passwd_submit_button:
                    if email_exists_check == False:
                        st.error("Email does not exist!")

                    elif current_passwd_check == False:
                        st.error("Incorrect temporary password!")

                    elif new_passwd != new_passwd_1:
                        st.error("Passwords don't match!")
                
                    if email_exists_check == True:
                        if current_passwd_check == True:
                            change_passwd(email_reset_passwd, new_passwd)
                            st.success("Password Reset Successfully!")
                    

        def logout_widget(self) -> None:
            """
            Creates the logout widget in the sidebar only if the user is logged in.
            """
            if st.session_state['LOGGED_IN'] == True:
                #del_logout = st.sidebar.empty()
                #del_logout.markdown("#")
                #logout_click_check = del_logout.button(self.logout_button_name)

                #if logout_click_check == True:
                st.session_state['LOGOUT_BUTTON_HIT'] = True
                st.session_state['LOGGED_IN'] = False
                self.cookies['__streamlit_login_signup_ui_username__'] = '1c9a923f-fb21-4a91-b3f3-5f18e3f0118212345'
                #del_logout.empty()
                st.experimental_rerun()
            

        def nav_sidebar(self):
            """
            Creates the side navigaton bar
            """
            main_page_sidebar = st.empty()
            with main_page_sidebar:
                selected_option = option_menu(
                    menu_title = '',
                    menu_icon = 'list-columns-reverse',
                    icons = ['house-door-fill','','','','','','person-fill-add', 'person-fill-add', 'x-circle','arrow-counterclockwise'],
                    orientation="horizontal",
                    default_index=0,
                    options = ['Home','---','---','---','---','---','Register'],
                    styles = {
                    "container": {"background-color": "#403f3f", "height": "10% !important" ,"width": "100%", "padding": "1px", "border-radius": "30px"},
                    "nav-link": {"padding": "5px", "height": "100%" ,"width": "100%" ,"font-weight": "bold","font-family": "Arial", "font-size": "65%", "color": "white", "text-align": "left", "margin":"0px",
                                    "--hover-opacity": "0%","text-align": "centre"},
                    "separator": {"opacity": "0% !important"},
                    "nav-link-selected": {
                    "padding": "5px",
                    "opacity":"100%",
                    "background": "#403f3f",
                    "font-family": "Arial",
                    "font-weight": "bold",
                    "font-size": "75%",
                    "color": "white",
                    "margin": "0px",
                    "position": "relative",
                    "overflow": "hidden",
                    "border-radius": "30px",
                    "text-decoration": "underline",
                    "font-weight": "bolder","text-align": "centre"

                    },
                    }  )
            return main_page_sidebar, selected_option
        

        def hide_menu(self) -> None:
            """
            Hides the streamlit menu situated in the top right.
            """
            st.markdown(""" <style>
            #MainMenu {visibility: hidden;}
            </style> """, unsafe_allow_html=True)
        

        def hide_footer(self) -> None:
            """
            Hides the 'made with streamlit' footer.
            """
            st.markdown(""" <style>
            footer {visibility: hidden;}
            </style> """, unsafe_allow_html=True)


        def build_login_ui(self):
            """
            Brings everything together, calls important functions.
            """
            
            if 'LOGGED_IN' not in st.session_state:
                st.session_state['LOGGED_IN'] = False

            if 'LOGOUT_BUTTON_HIT' not in st.session_state:
                st.session_state['LOGOUT_BUTTON_HIT'] = False

            auth_json_exists_bool = self.check_auth_json_file_exists('_secret_auth_.json')

            if auth_json_exists_bool == False:
                with open("_secret_auth_.json", "w") as auth_json:
                    json.dump([], auth_json)

            main_page_sidebar, selected_option = self.nav_sidebar()

            if selected_option == 'Signup/ Login':
                
                c1, c2 = st.columns([7,3])
                with c1:
                    self.login_widget()
                with c2:
                    if st.session_state['LOGGED_IN'] == False:
                        #self.animation()
                        print("")
                
                        
            if selected_option == 'Register':
                c1, c2 = st.columns([7,3])
                with c1:
                    
                    self.logingoogle_widget()
                    self.login_widget()
                    print("")
                with c2:
                    if st.session_state['LOGGED_IN'] == False:
                        #self.animation()
                        print("")
            if selected_option == 'Create Account':
                self.sign_up_widget()

            if selected_option == 'Forgot Password':
                self.forgot_password()
            if selected_option == 'Home':
             
                c1, c2,c3 = st.columns([5,8,5])
                with c2:
                    
                    self.logingoogle_widgetone()
                    
                    print("")
                with c3:
                    if st.session_state['LOGGED_IN'] == False:
                        #self.animation()
                        print("")
            
            #self.logout_widget()

            if st.session_state['LOGGED_IN'] == True:
                main_page_sidebar.empty()
            
            if self.hide_menu_bool == True:
                self.hide_menu()
            
            if self.hide_footer_bool == True:
                self.hide_footer()
            
            return st.session_state['LOGGED_IN']

    except Exception as e:
        print(e)
