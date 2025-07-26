import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  standalone: false,
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  username: string = '';
  password: string = '';
  email: string = '';
  forgotEmail: string = '';
  isLoginMode: boolean = true;
  isForgotPassword: boolean = false;

  constructor(private http: HttpClient, private router: Router) {}

  // Toggle between Login and Signup mode
  toggleMode() {
    this.isLoginMode = !this.isLoginMode;
    this.isForgotPassword = false;
    if (this.isLoginMode) {
      this.email = '';
    }
  }

  // Switch to Forgot Password mode
  toggleForgotPassword(){
    this.isForgotPassword= true;
    this.isLoginMode = false;
  }

  BacktoLogin(){
    this.isForgotPassword=false;
    this.isLoginMode=true;
  }

  // Handle Forgot Password submission
  onForgotPasswordSubmit() {
    const forgotPasswordData = { email: this.forgotEmail };
    const endpoint = 'http://127.0.0.1:5000/forgot-password';

    this.http.post(endpoint, forgotPasswordData , {withCredentials: true}).subscribe({
      next: (response: any) => {
        console.log('Forgot Password Successful:', response.message);
        alert('Password reset instructions have been sent to your email.');
        this.isForgotPassword= false; // Return to login mode
        this.forgotEmail = ''; // Clear input field
      },
      error: (err) => {
        console.error('Forgot Password Failed:', err);
        alert('Failed to process Forgot Password request. Please try again.');
      }
    });
  }

  onSubmit() {
    const userData = {
      username: this.username,
      password: this.password,
      email: this.email
    };

    const endpoint = this.isLoginMode
      ? 'http://127.0.0.1:5000/login'
      : 'http://127.0.0.1:5000/signup';

    this.http.post(endpoint, userData, { withCredentials: true }).subscribe({
      next: (response: any) => {
        console.log('Response from backend:', response);

        if (this.isLoginMode) {
          console.log('Login Successful:', response.message);
          localStorage.setItem('chat_id', response.chat_id);
        } else {
          console.log('Signup Successful:', response.message);
          localStorage.setItem('chat_id', '0');
        }

        this.router.navigate(['/home']);
      },
      error: (err) => {
        console.log('Request Failed:', err);
      }
    });
  }
}
