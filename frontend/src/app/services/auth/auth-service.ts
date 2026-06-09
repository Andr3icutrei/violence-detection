import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment.development';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { UserResponseDto } from '../../core/api/models/user-response-dto';
import { SocialAuthService } from '@abacritt/angularx-social-login';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  constructor(private http: HttpClient, private socialAuthService: SocialAuthService) {}

  public login(email: string, password: string): Observable<UserResponseDto> {
    if (!email || !password) {
      throw new Error('Email and password are required for registration.');
    }
    const body = {
      email: email,
      password: password,
    };
    return this.http.post<UserResponseDto>(environment.apiUrl + 'auth/login', body, { withCredentials: true });
  }

  public loginWithGoogle(tokenId: string): Observable<UserResponseDto> {
    const body = { tokenId };
    return this.http.post<UserResponseDto>(environment.apiUrl + 'auth/google-login', body, { withCredentials: true });
  }

  public logout(): Observable<void> {
    return this.http.post<void>(environment.apiUrl + 'auth/logout', null, { withCredentials: true }).pipe(
      tap({
        next: () => this.socialAuthService.signOut().catch(() => {}),
        error: () => this.socialAuthService.signOut().catch(() => {}),
      })
    );
  }

  public me(): Observable<UserResponseDto> {
    return this.http.get<UserResponseDto>(environment.apiUrl + 'auth/me', { withCredentials: true });
  }
}
