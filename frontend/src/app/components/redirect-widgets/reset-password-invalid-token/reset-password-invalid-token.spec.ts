import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ResetPasswordInvalidToken } from './reset-password-invalid-token';

describe('ResetPasswordInvalidToken', () => {
  let component: ResetPasswordInvalidToken;
  let fixture: ComponentFixture<ResetPasswordInvalidToken>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ResetPasswordInvalidToken],
    }).compileComponents();

    fixture = TestBed.createComponent(ResetPasswordInvalidToken);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
