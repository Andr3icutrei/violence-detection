import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { environment } from '../../../environments/environment.development';
import { Observable } from 'rxjs';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

@Injectable({
  providedIn: 'root',
})
export class UsersService {
  constructor(private httpClient: HttpClient) {}

  public register(email: string, password: string): Observable<UserResponseDto> {
    if (!email || !password) {
      throw new Error('Email and password are required for registration.');
    }
    const body = {
      email: email,
      password: password,
    };
    return this.httpClient.post<UserResponseDto>(environment.apiUrl + 'users/create', body);
  }

  public verifyAccount(token: string): Observable<UserResponseDto> {
    const params: HttpParams = new HttpParams().set('token', token);
    return this.httpClient.patch<UserResponseDto>(
      environment.apiUrl + 'users/verify_account',
      null,
      { params },
    );
  }

  public resendVerificationEmail(token: string): Observable<any> {
    const params: HttpParams = new HttpParams().set('token', token);

    return this.httpClient.get<any>(environment.apiUrl + 'users/resend_verification_email', {
      params,
    });
  }

  public verifyResetPasswordToken(token: string): Observable<any> {
    const params: HttpParams = new HttpParams().set('token', token);

    return this.httpClient.get<any>(environment.apiUrl + 'users/verify_reset_password_token', { params });
  }

  public requestResetPassword(email: string): Observable<any> {
    const params: HttpParams = new HttpParams().set('email', email);

    return this.httpClient.get<any>(environment.apiUrl + 'users/request_reset_password', {
      params,
    });
  }

  public resetPassword(token: string, newPassword: string): Observable<UserResponseDto> {
    const params: HttpParams = new HttpParams()
      .set('token', token)
      .set('newPassword', newPassword);

    return this.httpClient.patch<UserResponseDto>(environment.apiUrl + 'users/reset_password', null, { params },);
  }

  public getTopbarInformation(): Observable<UserResponseDto> {
    return this.httpClient.get<UserResponseDto>(environment.apiUrl + 'users/topbar_information', { withCredentials: true });
  }
}
